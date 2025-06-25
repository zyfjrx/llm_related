import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from dataset import get_dataloader
from model import ReviewAnalyzeModel
from tokenizer import JiebaTokenizer


def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, targets in tqdm(dataloader, desc='train'):
        inputs,targets = inputs.to(device),targets.to(device) # inputs:[batch_size,seq_len] targets:[batch_size]
        output = model(inputs) # output:[batch_size,1]
        loss = loss_fn(output, targets)
        total_loss += loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return total_loss / len(dataloader)



def train():
    device = torch.device('mps' if torch.mps.is_available() else 'cpu')

    # 加载词表
    tokenizer = JiebaTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'vocab.txt')
    model = ReviewAnalyzeModel(vocab_size=tokenizer.vocab_size,padding_idx=tokenizer.pad_token_id).to(device)
    dataloader = get_dataloader()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    train_one_epoch(dataloader, model, loss_fn, optimizer, device)
    write = SummaryWriter(log_dir=config.LOG_DIR / time.strftime("%Y%m%d-%H%M%S"))
    best_loss = float('inf')
    for epoch in range(1,config.EPOCHS+1):
        print(f"========== epoch {epoch} ==========")
        avg_loss = train_one_epoch(dataloader, model, loss_fn, optimizer, device)
        write.add_scalar('train/loss', avg_loss, epoch)
        print(f"avg_loss: {avg_loss}")
        if avg_loss < best_loss:
            print("误差减小了，保存模型 ...")
            best_loss = avg_loss
            torch.save(model.state_dict(), config.MODELS_DIR / 'model.pt')
        else:
            print("无需保存模型 ...")
    write.close()

if __name__ == '__main__':
    train()
