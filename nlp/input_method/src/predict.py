import jieba
import torch
from tokenizer import JiebaTokenizer
import config
from model import InputMethodModel
jieba.setLogLevel(jieba.logging.WARNING)

def predict_batch(model,inputs):
    with torch.no_grad():
        output = model(inputs) # output.shape [1,vocab_size]
        top5_indices = torch.topk(output, k=5).indices # top5_indices.shape [batch_size,5]
    top5_indexes_list = top5_indices.tolist()
    return top5_indexes_list

def predict(text,model,tokenizer,device):
    model.eval()
    input_ids = tokenizer.encode(text)
    input_ids = torch.tensor([input_ids]).to(device)  # input_ids.shape [1,seq_len]
    top5_indexes_list = predict_batch(model,input_ids)
    top5_words = tokenizer.decode(top5_indexes_list[0])
    return top5_words

def run_predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    # 加载词表
    tokenizer = JiebaTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'vocab.txt')
    model = InputMethodModel(vocab_size=tokenizer.vocab_size)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt'))
    model.to(device)

    print("请输入下一个词：(输入q或者quit推出)")
    history_input = ''
    while True:
        user_input = input("> ")
        if user_input in ['q', 'quit']:
            print("程序已退出")
            break
        if user_input.strip() == "":
            print("请输入下一个词：(输入q或者quit退出)")
            continue
        history_input += user_input
        print(f"历史输入：{history_input}")
        top5_words = predict(history_input,model,tokenizer,device)
        print(f"预测结果：{top5_words}")
if __name__ == "__main__":
    run_predict()
