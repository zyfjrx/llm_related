from torch.optim import AdamW
# 导入transformers库中的优化器调度器获取函数
from transformers.optimization import get_scheduler
import torch
import swanlab
from dataset import MyDataset
from transformers import AutoTokenizer
from transformers import GPT2LMHeadModel
swanlab.login(api_key="uWoxqhdeuw0t0XHV3wD7o")
swanlab.init(
  # 设置将记录此次运行的项目信息
  project="train_llm",
  workspace="aigc_zyf",
  experiment_name="gpt2_poem",
)
# 实例化自定义数据集
dataset = MyDataset()

tokenizer = AutoTokenizer.from_pretrained(r"/Users/zhangyf/PycharmProjects/train/llm_related/gpt2")
model = GPT2LMHeadModel.from_pretrained(r"/Users/zhangyf/PycharmProjects/train/llm_related/gpt2")


# 定义数据预处理函数，用于将文本编码成模型所需的格式
def collate_fn(data):
    # 使用分词器对数据进行编码，并添加必要的填充和截断
    data = tokenizer.batch_encode_plus(data,
                                       padding=True,  # 填充序列
                                       truncation=True,  # 截断序列
                                       max_length=512,  # 最大序列长度
                                       return_tensors='pt')  # 返回PyTorch张量

    # 创建标签，与输入ID相同
    data['labels'] = data['input_ids'].clone()

    return data

# 创建数据加载器，用于批量加载数据
loader = torch.utils.data.DataLoader(
    dataset=dataset,  # 指定数据集
    batch_size=6,  # 指定批量大小
    collate_fn=collate_fn,  # 指定预处理函数
    shuffle=True,  # 打乱数据
    drop_last=False,  # 如果最后一个批次不足，则丢弃
)

# 定义训练函数
def train():
    global model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    # 获取优化器调度器
    scheduler = get_scheduler(name='cosine',  # 线性调度器
                              num_warmup_steps=0,  # 预热步数
                              num_training_steps=len(loader),  # 总训练步数
                              optimizer=optimizer)

    # 设置模型为训练模式
    model.train()
    for i, data in enumerate(loader):
        for k in data.keys():
            data[k] = data[k].to(device)
        out = model(**data)
        loss = out['loss']
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        model.zero_grad()

        # 每隔50个批次打印一次训练信息
        if i % 50 == 0:
            # 准备标签和输出用于计算准确率
            labels = data['labels'][:, 1:]
            #通过‘logits’获取模型的原始输出值
            out = out['logits'].argmax(dim=2)[:, :-1]

            # 移除在数据预处理阶段添加的填充（通常是0），以便只计算实际数据部分的损失和准确率，避免填充部分对模型性能评估的影响。
            select = labels != 0
            labels = labels[select]
            out = out[select]
            del select

            # 计算准确率
            accuracy = (labels == out).sum().item() / labels.numel()

            # 获取当前学习率
            lr = optimizer.state_dict()['param_groups'][0]['lr']

            # 打印批次索引、损失、学习率和准确率
            swanlab.log({"loss": loss, "accuracy": accuracy, "lr": lr})
            print(i, loss.item(),lr, accuracy)
    # 保存模型参数，不保存模型结构
    # torch.save(model.state_dict(), 'net.pt')
    # print("权重保存成功！")
if __name__ == '__main__':
    for epoch in range(1000):
        train()
        # 保存模型参数，不保存模型结构
        torch.save(model.state_dict(), f'net_{epoch}.pt')
        print("权重保存成功！")
