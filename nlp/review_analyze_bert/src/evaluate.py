import torch
from tqdm import tqdm

import config
from dataset import get_dataloader
from transformers import AutoTokenizer,AutoModel
from model import ReviewAnalyzeModel
from predict import predict_batch

def evaluate(model, dataloader, device):
    total_count = 0
    correct_count = 0
    for batch in tqdm(dataloader, desc='evaluate'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['label']
        outputs = predict_batch(model, input_ids,attention_mask)
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
    tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_MODELS_DIR)
    model = ReviewAnalyzeModel()
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt'))
    model.to(device)

    dataloader = get_dataloader(train=False)
    acc = evaluate(model, dataloader, device)
    print("========== 评估结果 ==========")
    print(f"准确率: {acc:.4f}")
    print("=============================")

if __name__ == '__main__':
    run_evaluate()