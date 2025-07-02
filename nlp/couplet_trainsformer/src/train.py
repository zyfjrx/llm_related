import time
from transformers import AutoConfig

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from dataset import get_dataloader
from model import TranslationModel
from tokenizer import ChineseTokenizer


def train_one_epoch(train_dataloader, model, loss_fn, optimizer, device):
    model.train()
    epoch_total_loss = 0
    for inputs, targets in tqdm(train_dataloader, desc='train'):
        # inputs:[batch_size,seq_len] targets:[batch_size,seq_len]
        inputs, targets = inputs.to(device), targets.to(device)
        # decoder_input[batch_size,seq_len-1]
        decoder_input = targets[:, :-1]
        # decoder_target[batch_size,seq_len-1]
        decoder_target = targets[:, 1:]
        # src_pad_mask
        src_pad_mask = inputs.eq(model.src_embedding.padding_idx)
        tgt_mask = model.transformer.generate_square_subsequent_mask(decoder_input.size(1)).to(device)
        tgt_pad_mask = decoder_input.eq(model.tgt_embedding.padding_idx)
        # output[batch_size,seq_len-1,vocab_size]
        output = model(inputs, decoder_input,src_pad_mask, tgt_mask, tgt_pad_mask)

        # 计算损失
        loss = loss_fn(output.reshape(-1, output.shape[-1]), decoder_target.reshape(-1))
        epoch_total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return epoch_total_loss / len(train_dataloader)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 模型
    tokenizer = ChineseTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'vocab.txt')
    model =  TranslationModel(vocab_size=tokenizer.vocab_size,padding_idx=tokenizer.pad_token_id).to(device)

    # 训练数据
    train_dataloader = get_dataloader(train=True)

    # 损失函数、优化器
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.LEARNING_RATE)

    # tensorboard
    writer = SummaryWriter(log_dir=config.LOG_DIR / time.strftime("%Y%m%d-%H%M%S"))

    # 训练
    best_loss = float('inf')
    for epoch in range(1, config.EPOCHS + 1):
        print(f"========== epoch {epoch} ==========")
        avg_loss = train_one_epoch(train_dataloader, model, loss_fn, optimizer,device)
        writer.add_scalar('train/loss', avg_loss, epoch)
        print(f"avg_loss: {avg_loss}")
        if avg_loss < best_loss:
            print("误差减小了，保存模型 ...")
            best_loss = avg_loss
            torch.save(model.state_dict(), config.MODELS_DIR / 'model.pt')
            print("保存成功")
        else:
            print("无需保存模型 ...")


if __name__ == '__main__':
    train()
