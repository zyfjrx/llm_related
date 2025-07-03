import time

from torch.optim import AdamW
from transformers.optimization import get_scheduler
import torch
from dataset import get_dataloader
from transformers import AutoTokenizer
from transformers import GPT2LMHeadModel
from torch.utils.tensorboard import SummaryWriter
import config
# 实例化自定义数据集
dataloader = get_dataloader(train=True)

# 创建模型
tokenizer = AutoTokenizer.from_pretrained(r"/Users/zhangyf/llm/gpt2-chinese-cluecorpussmall")
model = GPT2LMHeadModel.from_pretrained(r"/Users/zhangyf/llm/gpt2-chinese-cluecorpussmall")


# 定义训练函数
def train(epochs):
    global model
    accumulation_steps = 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    # 获取优化器调度器
    iter_per_epoch = len(dataloader)
    total_steps = int(iter_per_epoch * epochs)
    scheduler = get_scheduler(name='cosine',  # 线性调度器
                              num_warmup_steps=0,  # 预热步数
                              num_training_steps=total_steps,  # 总训练步数
                              optimizer=optimizer)

    # 设置模型为训练模式
    write = SummaryWriter(log_dir=config.LOG_DIR / time.strftime("%Y%m%d-%H%M%S"))
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for step, data in enumerate(dataloader):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['input_ids'].to(device)
            out = model(input_ids=input_ids,labels=labels,attention_mask=attention_mask)
            # 梯度缩放
            loss = out['loss']
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            model.zero_grad()
            total_loss += loss.item()

        write.add_scalar('train/loss', total_loss/len(dataloader), epoch)

        # 保存模型参数，不保存模型结构

        torch.save(model.state_dict(), f'net_{epoch}.pt')
        print("权重保存成功！")


if __name__ == '__main__':
    epochs = 20
    train(epochs)
