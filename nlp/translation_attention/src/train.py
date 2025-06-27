import time
from itertools import chain

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from dataset import get_dataloader
from model import TranslationEncoder, TranslationDecoder
from tokenizer import ChineseTokenizer, EnglishTokenizer


def train_one_epoch(train_dataloader, encoder, decoder, loss_fn, optimizer, device):
    encoder.train()
    decoder.train()
    epoch_total_loss = 0
    for inputs, targets in tqdm(train_dataloader, desc='train'):
        # inputs:[batch_size,seq_len] targets:[batch_size,seq_len]
        inputs, targets = inputs.to(device), targets.to(device)
        # 编码
        # context_vector.shape[batch_size,decoder_hidden_size]
        encoder_outputs, context_vector = encoder(inputs)
        # 解码
        # decoder_input.shape: [batch_size, 1]
        decoder_input = targets[:, 0:1]
        # context_vector.shape[1,batch_size,decoder_hidden_size]
        decoder_hidden = context_vector.unsqueeze(0)
        decoder_outputs = []
        for t in range(1, targets.shape[1]):
            # decoder_output.shape: [batch_size, 1, vocab_size]
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs.append(decoder_output)
            decoder_input = targets[:, t:t + 1]
        # 预测结果
        # decoder_outputs.shape: [batch_size, seq_len - 1, vocab_size]
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        # decoder_outputs.shape: [batch_size*(seq_len - 1), vocab_size]
        decoder_outputs = decoder_outputs.reshape(-1, decoder_outputs.shape[-1])

        # 真实值
        # targets: [batch_size, seq_len-1]
        targets = targets[:, 1:]
        # targets: [batch_size*(seq_len - 1)]
        targets = targets.reshape(-1)

        # 计算损失
        loss = loss_fn(decoder_outputs, targets)
        epoch_total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return epoch_total_loss / len(train_dataloader)


def train():
    device = torch.device('mps' if torch.mps.is_available() else 'cpu')

    en_tokenizer = EnglishTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'vocab_en.txt')
    zh_tokenizer = ChineseTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'vocab_zh.txt')

    # 模型
    encoder = TranslationEncoder(vocab_size=zh_tokenizer.vocab_size,
                                 padding_idx=zh_tokenizer.pad_token_id).to(device)
    decoder = TranslationDecoder(vocab_size=en_tokenizer.vocab_size,
                                 padding_idx=en_tokenizer.pad_token_id).to(device)

    # 训练数据
    train_dataloader = get_dataloader(train=True)

    # 损失函数、优化器
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_id)
    optimizer = torch.optim.AdamW(params=chain(encoder.parameters(), decoder.parameters()), lr=config.LEARNING_RATE)

    # tensorboard
    writer = SummaryWriter(log_dir=config.LOG_DIR / time.strftime("%Y%m%d-%H%M%S"))

    # 训练
    best_loss = float('inf')
    for epoch in range(1, config.EPOCHS + 1):
        print(f"========== epoch {epoch} ==========")
        avg_loss = train_one_epoch(train_dataloader, encoder, decoder, loss_fn, optimizer, device)
        writer.add_scalar('train/loss', avg_loss, epoch)
        print(f"avg_loss: {avg_loss}")
        if avg_loss < best_loss:
            print("误差减小了，保存模型 ...")
            best_loss = avg_loss
            torch.save(encoder.state_dict(), config.MODELS_DIR / 'encoder_model.pt')
            torch.save(decoder.state_dict(), config.MODELS_DIR / 'decoder_model.pt')
            print("保存成功")
        else:
            print("无需保存模型 ...")


if __name__ == '__main__':
    train()
