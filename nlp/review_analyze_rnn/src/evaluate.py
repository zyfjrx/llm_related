import torch
from tqdm import tqdm

import config
from dataset import get_dataloader
from tokenizer import JiebaTokenizer
from model import ReviewAnalyzeModel
from predict import predict_batch

def evaluate(model, dataloader, device):
    total_count = 0
    correct_count = 0
    for inputs, targets in tqdm(dataloader, desc='evaluate'):
        inputs= inputs.to(device)
        outputs = predict_batch(model, inputs)
        for output,target in zip(outputs,targets):
            output = 1 if output > 0.5 else 0
            if output == target:
                correct_count += 1
            total_count += 1
    return correct_count / total_count


def run_evaluate():
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"设备: {device}")
    # 加载词表
    tokenizer = JiebaTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'vocab.txt')
    model = ReviewAnalyzeModel(vocab_size=tokenizer.vocab_size,padding_idx=tokenizer.pad_token_id)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt'))
    model.to(device)

    dataloader = get_dataloader(train=False)
    acc = evaluate(model, dataloader, device)
    print("========== 评估结果 ==========")
    print(f"准确率: {acc:.4f}")
    print("=============================")

if __name__ == '__main__':
    run_evaluate()