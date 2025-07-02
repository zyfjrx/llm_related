import jieba
import torch

import config
from tokenizer import JiebaTokenizer
from model import ReviewAnalyzeModel
from transformers import AutoTokenizer,AutoModel
jieba.setLogLevel(jieba.logging.WARNING)

def predict_batch(model, input_ids,attention_mask):
    model.eval()
    with torch.no_grad():
        output = model(input_ids,attention_mask) # output.shape [batch_size]
    return torch.sigmoid(output).tolist()


def predict(user_input, model, tokenizer, device):
    tokenized = tokenizer([user_input],max_length=config.SEQ_LEN,truncation=True,padding='max_length',return_tensors='pt')
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized['attention_mask'].to(device)
    batch_result = predict_batch(model, input_ids, attention_mask)
    return batch_result[0]


def run_predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_MODELS_DIR)
    model = ReviewAnalyzeModel()
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt'))
    model.to(device)

    print("请输入评价：(输入q或者quit推出)")
    while True:
        user_input = input("> ")
        if user_input in ['q', 'quit']:
            print("程序已退出")
            break
        if user_input.strip() == "":
            print("请输入评价：(输入q或者quit退出)")
            continue
        result = predict(user_input,model,tokenizer,device)
        if result > 0.5:
            print(f'正向评价（置信度：{result:.4f}）')
        else:
            print(f'负向评价（置信度：{1 - result:.4f}）')



if __name__ == '__main__':
    run_predict()