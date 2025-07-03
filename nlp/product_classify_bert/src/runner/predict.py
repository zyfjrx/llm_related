import torch
from models.bert_classifier import ProductClassifyModel
from conf import config
from transformers import AutoTokenizer
from preprocess.dataset import get_dataset, DataType


def predict_batch(model, input_ids, attention_mask):
    model.eval()
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        output = torch.argmax(output, dim=-1)
    return output.tolist()


def predict(user_input, model, tokenizer, device):
    tokenized = tokenizer([user_input],max_length=config.SEQ_LEN,truncation=True,padding='max_length',return_tensors='pt')
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized['attention_mask'].to(device)
    batch_result = predict_batch(model, input_ids, attention_mask)
    return batch_result[0]


def run_predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProductClassifyModel(freeze_bret=False).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR/'model.pt'))
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_MODELS_DIR)
    dataset = get_dataset(DataType.TRAIN)
    class_label
    print("========== 预测 ==========")
    print("请输入商品标题：(输入q或者quit退出)")
    while True:
        user_input = input("> ")
        if user_input in ['q', 'quit']:
            print("程序已退出")
            break
        if user_input.strip() == "":
            print("请输入商品标题：(输入q或者quit退出)")
            continue
        result = predict(user_input,model,tokenizer,device)
        print(result)

