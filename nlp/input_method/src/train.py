import time
import torch
from sympy.codegen.fnodes import cmplx
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dataset import get_dataloader
from model import InputMethodModel
import config


def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for input_tensor, target_tensor in tqdm(dataloader,desc='train'):
        input_tensor,target_tensor = input_tensor.to(device),target_tensor.to(device)
        output = model(input_tensor) # [batch_size,vocab_size]
        loss = loss_fn(output, target_tensor)
        total_loss += loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return total_loss / len(dataloader)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    dataloader = get_dataloader()
    print("数据集加载完成")

    # 加载词表
    with open(config.PROCESSED_DATA_DIR / 'vocab.txt', 'r', encoding='utf-8') as f:
        vocab_list = [line.strip() for line in f.readlines()]
    model = InputMethodModel(vocab_size=len(vocab_list)).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_loss = float('inf')
    writer = SummaryWriter(log_dir=config.LOG_DIR / time.strftime("%Y%m%d-%H%M%S"))
    for epoch in range(1,config.EPOCHS+1):
        print(f"========== epoch {epoch} ==========")
        avg_loss = train_one_epoch(dataloader, model, loss_fn, optimizer, device)
        writer.add_scalar('train/loss', avg_loss, epoch)
        print(f"avg_loss: {avg_loss}")
        if avg_loss < best_loss:
            print("误差减小了，保存模型 ...")
            best_loss = avg_loss
            torch.save(model.state_dict(), config.MODELS_DIR / 'model.pt')
        else:
            print("无需保存模型 ...")
    writer.close()


if __name__ == '__main__':
    train()