import time

from tqdm import tqdm

from conf import config
import torch
from models.bert_classifier import ProductClassifyModel
from preprocess.dataset import DataType,get_dataloader
from torch.utils.tensorboard import SummaryWriter


def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    model.train()
    epoch_total_loss = 0
    for data in tqdm(dataloader, desc='train'):
        # inputs:[batch_size,seq_len] labels:[batch_size]
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['label'].to(device)
        # outputs.shape[batch_size,num_class]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # 获取损失
        loss = loss_fn(outputs, labels)
        epoch_total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return epoch_total_loss / len(dataloader)


def train():
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    dataloader = get_dataloader(type=DataType.TRAIN)
    model = ProductClassifyModel(freeze_bret=False).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

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
            torch.save(model.state_dict(), config.MODELS_DIR / 'best.pt')
        else:
            print("无需保存模型 ...")


