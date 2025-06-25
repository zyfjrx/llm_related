import jieba
import torch

import config
from tokenizer import JiebaTokenizer
from model import ReviewAnalyzeModel

jieba.setLogLevel(jieba.logging.WARNING)

def predict_batch(model, input_ids):
    """
    批量预测
    :param model:
    :param input_ids: 输入张量:[batch_size,seq_len]
    :return:
    """
    model.eval()
    with torch.no_grad():
        output = model(input_ids) # output.shape [batch_size]
    return torch.sigmoid(output).tolist()


def predict(user_input, model, tokenizer, device):
    input_ids = tokenizer.encode(user_input,config.SEQ_LEN)
    input_ids = torch.tensor([input_ids]).to(device)
    batch_result = predict_batch(model, input_ids)
    return batch_result[0]


def run_predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    tokenizer = JiebaTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'vocab.txt')
    model = ReviewAnalyzeModel(vocab_size=tokenizer.vocab_size,padding_idx=tokenizer.pad_token_id)
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