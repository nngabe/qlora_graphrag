import gdown

url = "https://drive.google.com/drive/folders/1yNe4-rBckupzdnPQGEqlHBsENAuPPLd2"
gdown.download_folder(url, quiet=False, use_cookies=False)
