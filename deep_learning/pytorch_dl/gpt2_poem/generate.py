import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
device = torch.device('mps' if torch.mps.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(constants.GPT2_PATH)
model = AutoModelForCausalLM.from_pretrained(constants.GPT2_PATH)

# 加载自己训练的模型权重
model.load_state_dict(torch.load("../trained_models/gpt2/net.pt", map_location=device))
model.to(device)


# 定制化pipline工具生成五言绝句内容
# text提示词、row是生成文本行数、col是每行的字符数
def generate(text, row, col):
    # 定义一个内部递归函数用于生成文本
    def generate_loop(data):
        # 禁用梯度计算
        with torch.no_grad():
            # 使用data中的字典数据作为输入，并获得输出
            out = model(**data)
        #获取最后一个字(logits未归一化的概率输出)
        out = out["logits"]
        print("logits",out.shape)
        #选择每个序列的最后一个logits，对应于下一个词的预测
        out = out[:,-1]
        print("output",out.shape)
        # 找到概率排名前50的值，以此为分界线，小于该值的全部舍去
        topk_value = torch.topk(out, 50).values
        # 获取每个输出序列中第50个logits的值（为保证原维度，需要对结果增加一个维度，因为索引操作会降维度）
        topk_value = topk_value[:, -1].unsqueeze(dim=1)
        # 将所有小于第50个的值的logits设置为负无穷，减少概率词被选中的可能性
        out = out.masked_fill(out < topk_value, -float('inf'))
        # 屏蔽掉[UNK]
        out[:, tokenizer.get_vocab()["[UNK]"]] = -float("inf")
        # 将特殊词标点符号设置为负无穷，减少概率词被选中的可能性
        for i in ",.()《》[]「」{}":
            out[:, tokenizer.get_vocab()[i]] = -float('inf')

        # 根据概率采样，无放回避免生成重复的内容
        out = out.softmax(dim=1)
        # 从概率分布中随机采样，选择下一个词ID
        out = out.multinomial(num_samples=1)
        print("output2",out.shape)

        # 强制添加标点符号
        # 计算当前生成文本长度与预期的长度的比例

        c = data["input_ids"].shape[1] / (col + 1)
        if c % 1 == 0:
            if c % 2 == 0:
                out[:, 0] = tokenizer.get_vocab()["."]
            else:
                out[:, 0] = tokenizer.get_vocab()[","]



        data["input_ids"] = torch.cat([data["input_ids"], out], dim=1)
        data["attention_mask"] = torch.ones_like(data["input_ids"])
        data["token_type_ids"] = torch.zeros_like(data["input_ids"])
        data["labels"] = data["input_ids"].clone()

        # 检查生成的文本是否符合要求
        if data["input_ids"].shape[1] >= row*col + row+1:
            print("input_ids", data["input_ids"].shape)
            return data
        else:
            return generate_loop(data)

    # 生成三首诗词
    # 使用tokenizer对输入文本进行编码，并重复3次生成3个样本
    data = tokenizer.batch_encode_plus([text]*3, return_tensors='pt')
    # 移除编码后的序列中的最后一个token（结束符号）
    data['input_ids'] = data['input_ids'][:,:-1]
    data['attention_mask'] = torch.ones_like(data['input_ids'])
    data['token_type_ids'] = torch.zeros_like(data['input_ids'])
    data['labels'] = data['input_ids'].clone()
    data.to(device)
    data = generate_loop(data)

    for i in range(3):
        print(i,tokenizer.decode(data["input_ids"][i]))

if __name__ == '__main__':
    generate("天",row=4,col=5)
