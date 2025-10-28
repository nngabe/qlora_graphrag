import argparse
import math
import os
import gc
import time
from tqdm import tqdm

import accelerate
import torch
import transformers

from dotenv import load_dotenv
from torch_geometric.loader import DataLoader

from stark_qa import load_qa
from torch import Tensor
from torch_geometric import seed_everything
from torch_geometric.nn.models import GAT

from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

from LLM import LLM
from GRetriever import GRetriever

from gretriever.compute_metrics import compute_metrics

from gretriever.STaRKQADatasetGDS import STaRKQADataset
from gretriever.STaRKQAVectorSearchDataset import STaRKQAVectorSearchDataset

llama_dict = {'llama3.2-1b': 'meta-llama/Llama-3.2-1B-Instruct',
              'llama3.2-3b': 'meta-llama/Llama-3.2-3B-Instruct',
              'llama3.1-8b': 'meta-llama/Llama-3.1-8B-Instruct',
              'llama3.3-70b': 'meta-llama/Llama-3.3-70B-Instruct'}
 


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
        return model(batch.question, batch.x, batch.edge_index, batch.batch,
                     batch.label, batch.edge_attr, batch.desc, inference=True)


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
    accelerator=None,
    attn_implementation='eager',
    max_seq_len=None,
):

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
        model_dir = root_path+'/models'
        train_dataset = STaRKQADataset(root_path, qa_raw_train, retrieval_config_version, algo_config_version, split="train")
        print(f'Finished loading train dataset in {time.time() - t} seconds.')
        print("Loading stark-qa prime val dataset...")
        val_dataset = STaRKQADataset(root_path, qa_raw_val, retrieval_config_version, algo_config_version, split="val")
        print("Loading stark-qa prime test dataset...")
        test_dataset = STaRKQADataset(root_path, qa_raw_test, retrieval_config_version, algo_config_version, split="test")
        os.makedirs(f'{root_path}/models', exist_ok=True)
   
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              drop_last=True, pin_memory=False, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size,
                            drop_last=False, pin_memory=False, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size,
                             drop_last=False, pin_memory=False, shuffle=False)

    gnn = GAT(
        in_channels=1536,
        hidden_channels=hidden_channels,
        out_channels=1536,
        num_layers=num_gnn_layers,
        heads=4,
    )
    
    #gnn = gnn.to(torch.bfloat16)

    if not use_quantization:
        quantization_config=None
    print(f'\n use_quantization={use_quantization}\n quantization_config={quantization_config}\n')
    
    if not use_lora:
        lora_config=None
    print(f'\n use_lora={use_lora}\n lora_config={lora_config}\n')
    
    llm = LLM(
            model_name=llama_dict[llama_version],
            quantization_config=quantization_config,
            lora_config=lora_config,
            accelerator=accelerator,
            attn_implementation=attn_implementation,
            max_seq_len=max_seq_len,
        )


    if args.freeze_llm:
        print(f'freeze_llm={args.freeze_llm}, freezing llm... \n')
        for param in llm.parameters():
            param.requires_grad = False
    

    if model_save_name == f'llm-{llama_version}':
        model = llm
    else:
        model = GRetriever(llm=llm, gnn=gnn, use_lora=use_lora, lora_config=lora_config, accelerator=accelerator)

    
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {
            'params': params,
            'lr': lr,
            'weight_decay': 0.05
        },
    ], betas=(0.9, 0.95))
    grad_steps = 2

    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader)*num_epochs,
    )

    print(f"Model parameters before prepare: {set(p.device.type for p in model.parameters())}")
    model, optimizer = accelerator.prepare(model, optimizer)
    scheduler = accelerator.prepare(scheduler)
    train_loader = accelerator.prepare(train_loader)
    val_loader = accelerator.prepare(val_loader)
    test_loader = accelerator.prepare(test_loader)

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
            optimizer.zero_grad()
            loss = get_loss(model, batch, model_save_name)
            accelerator.backward(loss)
            if step%10 == 0:
                ewma = loss.item() if step==0 else .9 * ewma + .1 * loss.item()
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
                metrics = {'step': step, 'loss': ewma, 'grad_norm': total_norm.item(), 'lr': optimizer.param_groups[0]['lr']}
                tqdm.write(str(metrics))
            accelerator.clip_grad_norm_(model.parameters(), max_norm=0.1)   

            optimizer.step()
            scheduler.step()
            epoch_loss = epoch_loss + float(loss)
            
            if (step%500)==0:
                if torch.cuda.is_available():
                    rank = int(str(accelerator.device)[-1])
                    print(f'max_memory_allocated[{accelerator.device}]({step}/{len(train_loader)}): {torch.cuda.max_memory_allocated()/10**9:.2f} GB')

        train_loss = epoch_loss / len(train_loader)
        print(epoch_str + f', Train Loss: {train_loss:4f}')

        gc.collect()
        torch.cuda.empty_cache()
        
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                loss = get_loss(model, batch, model_save_name)
                val_loss += loss.item()
            val_loss = val_loss / len(val_loader)
            print(epoch_str + f", Val Loss: {val_loss:4f}")
        if checkpointing and val_loss < best_val_loss:
            print("Checkpointing best model...")
            best_val_loss = val_loss
            best_epoch = epoch
            accelerator.save_model(model, model_dir)
            accelerator.load_state(model_dir)

    if checkpointing and best_epoch != num_epochs - 1:
        print("Loading best checkpoint...")
        accelerator.load_state(model_dir)

    model.eval()
    eval_output = []
    print("Final evaluation...")
    progress_bar_test = tqdm(range(len(test_loader)))
    for step, batch in enumerate(test_loader):
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
    accelerator.save_model(model, root_path)
    torch.save(eval_output, f'{root_path}/models/{time.time()}_{model_save_name}_eval_outs.pt')


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--gnn_hidden_channels', type=int, default=1536)
    parser.add_argument('--num_gnn_layers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--checkpointing', action='store_true')
    parser.add_argument('--llama_version', type=str, required=True)
    parser.add_argument('--retrieval_config_version', type=int, default=0)
    parser.add_argument('--algo_config_version', type=int, default=0)
    parser.add_argument('--g_retriever_config_version', type=int, default=0)
    parser.add_argument('--freeze_llm', action='store_true') 
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--attn_implementation', type=str, default='flash_attention_2' if torch.backends.cuda.flash_sdp_enabled() else 'eager')
    parser.add_argument('--use_quantization', action='store_true')
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--init_lora_weights', type=str, default=True)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--max_seq_len', type=int, default=2048)

    args = parser.parse_args()
    load_dotenv('db.env', override=True)

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    args.device = accelerator.device

    lora_config = LoraConfig(
        init_lora_weights=args.init_lora_weights,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
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
        accelerator=accelerator,
        attn_implementation=args.attn_implementation,
        max_seq_len=args.max_seq_len,
    )
    print(f"Total Time: {time.time() - start_time:2f}s")

