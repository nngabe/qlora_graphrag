import argparse
import math
import os
import time

import torch
from dotenv import load_dotenv
from torch_geometric.loader import DataLoader

from stark_qa import load_qa
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch_geometric import seed_everything
from torch_geometric.nn.models import GAT
from peft import LoraConfig
from transformers import BitsAndBytesConfig
import bitsandbytes as bnb
from tqdm import tqdm

from gretriever.MPNN import MPNN

from gretriever.LLM import LLM
from gretriever.GRetriever import GRetriever

from gretriever.compute_metrics import compute_metrics

from gretriever.STaRKQADatasetGDS import STaRKQADataset
from gretriever.STaRKQAVectorSearchDataset import STaRKQAVectorSearchDataset

def count_parameters(models):
    if type(models)!=list:
        models = [models]
    count = 0
    for model in models:
        count += sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f'{str(count//1e+6)}M'

def get_loss(model, batch, model_save_name) -> Tensor:
    if model_save_name.startswith('llm'):
        return model(batch.question, batch.label, batch.desc)
    else:
        # calls forward for GRetriever
        return model(batch.question, batch.x, batch.edge_index, batch.batch,
                     batch.label, batch.edge_attr, batch.desc)


def inference_step(model, batch, model_save_name):
    if model_save_name.startswith('llm'):
        return model.inference(batch.question, batch.desc)
    else:
        return model.inference(batch.question, batch.x, batch.edge_index,
                               batch.batch, batch.edge_attr, batch.desc)

def save_params_dict(model, save_path):
    state_dict = model.state_dict()
    param_grad_dict = {
        k: v.requires_grad
        for (k, v) in model.named_parameters()
    }
    for k in list(state_dict.keys()):
        if k in param_grad_dict.keys() and not param_grad_dict[k]:
            del state_dict[k]  # Delete parameters that do not require gradient
    torch.save(state_dict, save_path)

def load_params_dict(model, save_path):
    state_dict = model.state_dict()
    state_dict.update(torch.load(save_path)) #All weights might not be saved, eg when using LoRA.
    model.load_state_dict(state_dict)
    return model


def train(
    num_epochs,
    hidden_channels,
    num_gnn_layers,
    batch_size,
    eval_batch_size,
    lr,
    llama_version,
    retrieval_config_version,
    algo_config_version,
    g_retriever_config_version,
    use_lora,
    use_quantization,
    lora_config,
    quantization_config,
    checkpointing=False,
    sys_prompt=None,
    num_gpus=None,
    args=None
):
    def adjust_learning_rate(param_group, LR, epoch):
        # Decay the learning rate with half-cycle cosine after warmup
        min_lr = 5e-6
        warmup_epochs = 1
        if epoch < warmup_epochs:
            lr = LR
        else:
            lr = min_lr + (LR - min_lr) * 0.5 * (
                    1.0 + math.cos(math.pi * (epoch - warmup_epochs) /
                                   (num_epochs - warmup_epochs)))
        param_group['lr'] = lr
        return lr

    start_time = time.time()
    qa_dataset = load_qa("prime")
    qa_raw_train = qa_dataset.get_subset('train')
    qa_raw_val = qa_dataset.get_subset('val')
    qa_raw_test = qa_dataset.get_subset('test')
    seed_everything(42)

    print("Loading stark-qa prime train dataset...")
    t = time.time()

    if num_gnn_layers == 0:
        model_save_name = f'llm-{llama_version}'
    else:
        if args.freeze_llm:
            model_save_name = f'gnn-frozen-llm-{llama_version}'
        else:
            model_save_name = f'gnn-llm-{llama_version}'

    if model_save_name == f'llm-{llama_version}':
        root_path = f"stark_qa_vector_rag_{retrieval_config_version}"
        train_dataset = STaRKQAVectorSearchDataset(root_path, qa_raw_train, split="train")
        print(f'Finished loading train dataset in {time.time() - t} seconds.')
        print("Loading stark-qa prime val dataset...")
        val_dataset = STaRKQAVectorSearchDataset(root_path, qa_raw_val, split="val")
        print("Loading stark-qa prime test dataset...")
        test_dataset = STaRKQAVectorSearchDataset(root_path, qa_raw_test, split="test")
        os.makedirs(f'{root_path}/models', exist_ok=True)
    else:
        root_path = f"stark_qa_v{retrieval_config_version}_{algo_config_version}"
        train_dataset = STaRKQADataset(root_path, qa_raw_train, retrieval_config_version, algo_config_version, split="train")
        print(f'Finished loading train dataset in {time.time() - t} seconds.')
        print("Loading stark-qa prime val dataset...")
        val_dataset = STaRKQADataset(root_path, qa_raw_val, retrieval_config_version, algo_config_version, split="val")
        print("Loading stark-qa prime test dataset...")
        test_dataset = STaRKQADataset(root_path, qa_raw_test, retrieval_config_version, algo_config_version, split="test")
        os.makedirs(f'{root_path}/models', exist_ok=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              drop_last=True, pin_memory=False, shuffle=True,
                              generator=torch.Generator(device=torch.get_default_device().type))
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size,
                            drop_last=False, pin_memory=False, shuffle=False,
                            generator=torch.Generator(device=torch.get_default_device().type))
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size,
                             drop_last=False, pin_memory=False, shuffle=False,
                             generator=torch.Generator(device=torch.get_default_device().type))
   
    if args.gnn=='gat':
        gnn = GAT(
            in_channels=1536,
            hidden_channels=6144,
            out_channels=2048,
            num_layers=6,
            heads=8,
        )
    elif args.gnn=='mpnn':
        gnn = MPNN(
            in_channels=1536,
            hidden_channels=2048,
            out_channels=2048,
            num_layers=4,
            aggr=['sum','max','mean','var'],
            dropout=0.05,
            ffw_dim=2,
        )
    elif args.gnn=='gat_big':
        gnn = GAT(
            in_channels=1536,
            hidden_channels=8192,
            out_channels=1536,
            num_layers=6,
            heads=16,
        ).to(torch.bfloat16)
    elif args.gnn=='mpnn_big':
        gnn = MPNN(
            in_channels=1536,
            hidden_channels=2048,
            out_channels=1536,
            num_layers=4,
            aggr=['sum','max','mean','var'],
            dropout=0.05,
            ffw_dim=2,
        ).to(torch.bfloat16)

    gnn = gnn.to(torch.get_default_device().type)


    if not use_quantization:
        quantization_config=None
    print(f'\n use_quantization={use_quantization}\n quantization_config={quantization_config}\n')
    
    if not use_lora:
        lora_config=None
    print(f'\n use_lora={use_lora}\n lora_config={lora_config}\n')
    
    if llama_version == 'llama3.1-8b':
        llm = LLM(
            model_name='meta-llama/Llama-3.1-8B-Instruct',
            quantization_config=quantization_config,
        )
    elif llama_version == 'llama3.2-1b':
        llm = LLM(
            model_name='meta-llama/Llama-3.2-1B-Instruct',
            quantization_config=quantization_config,
        )
    elif llama_version == 'llama3.2-3b':
        llm = LLM(
            model_name='meta-llama/Llama-3.2-3B-Instruct',
            quantization_config=quantization_config,
        )
    elif llama_version == 'llama3.3-70b':
        llm = LLM(
            model_name='meta-llama/Llama-3.3-70B-Instruct',
            quantization_config=quantization_config,
        )
    
    llm.device = torch.get_default_device()
    llm = llm.to(torch.get_default_device().type)
    llm.llm = llm.llm.to(torch.get_default_device().type)
    llm.word_embedding = llm.llm.model.get_input_embeddings()
    print(f'\nllm.word_embedding(1)={llm.word_embedding(torch.tensor(1))}\n')


    if args.freeze_llm:
        print(f'freeze_llm={args.freeze_llm}, freezing llm... \n')
        for param in llm.parameters():
            param.requires_grad = False

    if model_save_name == f'llm-{llama_version}':
        model = llm
    else:
        model = GRetriever(llm=llm, gnn=gnn, use_lora=use_lora, lora_config=lora_config, gnn_out_tokens=args.gnn_out_tokens)
    
    print(model.gnn)
    print(model.projector)
    print(model.llm_generator)


    print(f'\n gnn trainable params: {count_parameters([model.gnn])}')
    print(f'\n proj. trainable params: {count_parameters([model.projector])}')
    print(f'\n llm trainable params: ',end='') 
    model.llm_generator.print_trainable_parameters()
    
    print(f"\nModel device is: {llm.device}\n")

    params = [p for _, p in model.named_parameters() if p.requires_grad]
    if args.paged_adamw:
        optimizer = bnb.optim.PagedAdamW8bit(params=params, lr=lr, betas=(0.9, 0.995), weight_decay=0.05)
    else:
        optimizer = bnb.optim.AdamW8bit(params=params, lr=lr, betas=(0.9, 0.995), weight_decay=0.05)
    grad_steps = 2

    print(f'optimizer: {optimizer}\n')

    best_epoch = 0
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        if epoch == 0:
            print(f"Total Preparation Time: {time.time() - start_time:2f}s")
            start_time = time.time()
            print("Training beginning...")
        epoch_str = f'Epoch: {epoch + 1}|{num_epochs}'
        loader = tqdm(train_loader, desc=epoch_str)
        
        if torch.cuda.is_available():
            torch.cuda.reset_max_memory_allocated()

        for step, batch in enumerate(loader):
            batch = batch.to(torch.get_default_device().type)
            optimizer.zero_grad()
            loss = get_loss(model, batch, model_save_name)
            loss.backward()
            if step%10 == 0:
                ewma = loss.item() if step==0 else .9 * ewma + .1 * loss.item() 
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
                metrics = {'step': step, 'loss': ewma, 'grad_norm': total_norm.item(), 'lr': optimizer.param_groups[0]['lr']}
                tqdm.write(str(metrics))
            clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)

            if (step + 1) % grad_steps == 0:
                adjust_learning_rate(optimizer.param_groups[0], lr,
                                     step / len(train_loader) + epoch)
            optimizer.step()
            epoch_loss = epoch_loss + float(loss)
            
            if (step%100)==0:
                if torch.cuda.is_available():
                    print(f'max_memory_allocated({step}/{loader.total}): {torch.cuda.max_memory_allocated()/10**9:.2f} GB')

            if (step + 1) % grad_steps == 0:
                lr = optimizer.param_groups[0]['lr']

        train_loss = epoch_loss / len(train_loader)
        print(epoch_str + f', Train Loss: {train_loss:4f}')


        val_loss = 0
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                batch = batch.to(torch.get_default_device().type)
                loss = get_loss(model, batch, model_save_name)
                val_loss += loss.item()
            val_loss = val_loss / len(val_loader)
            print(epoch_str + f", Val Loss: {val_loss:4f}")
        if checkpointing and val_loss < best_val_loss:
            print("Checkpointing best model...")
            best_val_loss = val_loss
            best_epoch = epoch
            save_params_dict(model, f'{root_path}/models/{retrieval_config_version}_{algo_config_version}_{g_retriever_config_version}_{model_save_name}_best_val_loss_ckpt.pt')
    

    if checkpointing and best_epoch != num_epochs - 1:
        print("Loading best checkpoint...")
        model = load_params_dict(
            model,
            f'{root_path}/models/{retrieval_config_version}_{algo_config_version}_{g_retriever_config_version}_{model_save_name}_best_val_loss_ckpt.pt',
        )

    model.eval()
    eval_output = []
    print("Final evaluation...")
    progress_bar_test = tqdm(range(len(test_loader)))
    for step, batch in enumerate(test_loader):
        batch = batch.to(torch.get_default_device().type)
        with torch.no_grad():
            pred_time = time.time()
            pred = inference_step(model, batch, model_save_name)
            print(f"Time to predict: {time.time() - pred_time:2f}s")
            eval_data = {
                'pred': pred,
                'question': batch.question,
                'desc': batch.desc,
                'label': batch.label
            }
            eval_output.append(eval_data)
        progress_bar_test.update(1)

    compute_metrics(eval_output)
    print(f"Total Training Time: {time.time() - start_time:2f}s")
    save_params_dict(model, f'{root_path}/models/{retrieval_config_version}_{algo_config_version}_{g_retriever_config_version}_{model_save_name}.pt')
    torch.save(eval_output, f'{root_path}/models/{retrieval_config_version}_{algo_config_version}_{g_retriever_config_version}_{model_save_name}_eval_outs.pt')


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--gnn_hidden_channels', type=int, default=1536)
    parser.add_argument('--num_gnn_layers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--checkpointing', action='store_true')
    parser.add_argument('--gnn', type=str, default='gat')
    parser.add_argument('--gnn_out_tokens', type=int, default=8)
    parser.add_argument('--llama_version', type=str, required=True)
    parser.add_argument('--retrieval_config_version', type=int, default=0)
    parser.add_argument('--algo_config_version', type=int, default=0)
    parser.add_argument('--g_retriever_config_version', type=int, default=0)
    parser.add_argument('--freeze_llm', action='store_true') 
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--use_quantization', action='store_true')
    parser.add_argument('--paged_adamw', action='store_true')
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=-1)
    parser.add_argument('--init_lora_weights', type=str, default=True)
    parser.add_argument('--device', type=str, required=True)
    args = parser.parse_args()
    load_dotenv('db.env', override=True)

    if args.lora_alpha==-1:
        args.lora_alpha = args.lora_rank//2

    torch.set_default_device(args.device)


    lora_config = LoraConfig(
        init_lora_weights=args.init_lora_weights,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    start_time = time.time()
    train(
        args.epochs,
        args.gnn_hidden_channels,
        args.num_gnn_layers,
        args.batch_size,
        args.eval_batch_size,
        args.lr,
        llama_version=args.llama_version,
        retrieval_config_version=args.retrieval_config_version,
        algo_config_version=args.algo_config_version,
        g_retriever_config_version=args.g_retriever_config_version,
        use_lora=args.use_lora,
        use_quantization=args.use_quantization,
        lora_config=lora_config,
        quantization_config=quantization_config,
        checkpointing=args.checkpointing,
        sys_prompt=None,
        num_gpus=None,
        args=args,
    )
    print(f"Total Time: {time.time() - start_time:2f}s")

