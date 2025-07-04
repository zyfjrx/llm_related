import torch
from tqdm import tqdm

from conf import config
from preprocess.dataset import get_dataloader,get_dataset,DataType
from transformers import AutoTokenizer,AutoModel
from models.bert_classifier import ProductClassifyModel
from runner.predict import predict_batch

def evaluate(model, dataloader, device):
    total_count = 0
    correct_count = 0
    for batch in tqdm(dataloader, desc='evaluate'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['label']
        outputs = predict_batch(model, input_ids,attention_mask)

        correct_count += (torch.tensor(outputs) == targets).sum().item()
        total_count += len(targets)
    return correct_count / total_count


def run_evaluate():
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"设备: {device}")
    # 加载词表
    tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_MODELS_DIR)
    model = ProductClassifyModel()
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pt'))
    model.to(device)

    dataloader = get_dataloader(DataType.TEST)
    acc = evaluate(model, dataloader, device)
    print("========== 评估结果 ==========")
    print(f"准确率: {acc:.4f}")
    print("=============================")

if __name__ == '__main__':
    run_evaluate()