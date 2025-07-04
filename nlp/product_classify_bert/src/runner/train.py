import time

from tqdm import tqdm

from conf import config
import torch
from models.bert_classifier import ProductClassifyModel
from preprocess.dataset import DataType, get_dataloader
from torch.utils.tensorboard import SummaryWriter


class EarlyStopping:
    def __init__(self, patience=2, path=None):
        """
        初始化早停机制
        :param patience: 容忍验证集loss连续几轮不下降（超过立即停止）
        :param path: 模型保存路径
        """
        self.patience = patience
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
            self.save_model(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_model(self, model):
        torch.save(model.state_dict(), self.path)


def run_one_epoch(dataloader, model, loss_fn, optimizer, device, scaler,is_train=True):
    model.train() if is_train else model.eval()
    epoch_total_loss = 0
    with torch.set_grad_enabled(is_train):
        for data in tqdm(dataloader, desc='train'if is_train else 'valid'):
            # inputs:[batch_size,seq_len] labels:[batch_size]
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['label'].to(device)
            # outputs.shape[batch_size,num_class]
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs, labels)
            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            epoch_total_loss += loss.item()
    return epoch_total_loss / len(dataloader)


def train():
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    train_dataloader = get_dataloader(type=DataType.TRAIN)
    valid_dataloader = get_dataloader(type=DataType.VALID)
    model = ProductClassifyModel(freeze_bret=False).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # 初始化梯度缩放
    scaler = torch.GradScaler()
    # 早停机制
    early_stop = EarlyStopping(patience=3,path=config.MODELS_DIR / 'best.pt')

    write = SummaryWriter(log_dir=config.LOG_DIR / time.strftime("%Y%m%d-%H%M%S"))

    # 检查点配置
    start_epoch = 1
    checkpoint_path = config.MODELS_DIR / 'checkpoint.pt'
    if checkpoint_path.exists():
        print("检查点已存在，开始恢复训练 ...")
        checkpoint = torch.load(checkpoint_path,map_location= device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        early_stop.best_score = checkpoint['best_score']
        early_stop.counter = checkpoint['counter']
        start_epoch = checkpoint['epoch'] + 1
        print(f"已恢复训练，当前epoch为：{start_epoch}")
    else:
        print("检查点不存在，开始训练 ...")


    for epoch in range(start_epoch, config.EPOCHS + 1):
        print(f"========== epoch {epoch} ==========")
        # 训练阶段
        train_loss = run_one_epoch(train_dataloader, model, loss_fn, optimizer, device, scaler, is_train=True)
        # 验证阶段
        valid_loss = run_one_epoch(valid_dataloader, model, loss_fn, optimizer, device, scaler, is_train=False)
        write.add_scalar('train/loss', train_loss, epoch)
        write.add_scalar('valid/loss', valid_loss, epoch)
        print(f"train_loss: {train_loss}")
        print(f"valid_loss: {valid_loss}")
        early_stop(valid_loss, model)
        if early_stop.early_stop:
            print("已满足早停条件，停止训练 ...")
            break
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_score': early_stop.best_score,
            'counter': early_stop.counter
        }
        torch.save(checkpoint, checkpoint_path)
    write.close()

