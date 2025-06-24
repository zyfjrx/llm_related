import torch
from tqdm import tqdm
from predict import predict_batch
import config
from model import InputMethodModel
from dataset import get_dataloader


def evaluate_model(model, dataloader, device):
    total_count = 0
    top1_acc_count = 0
    top5_acc_count = 0
    for inputs, targets in tqdm(dataloader, desc='evaluate'):
        inputs = inputs.to(device)
        targets = targets.tolist() # [batch_size]  eg.[5,8,13]
        top5_indexes_list = predict_batch(model, inputs)
        # top5_indexes_list = [[],[],[]]
        for target, top5_indexes in zip(targets, top5_indexes_list):
            total_count += 1
            if target in top5_indexes:
                top5_acc_count += 1
            if target == top5_indexes[0]:
                top1_acc_count += 1
    return top1_acc_count / total_count, top5_acc_count / total_count

def run_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    # 加载词表
    with open(config.PROCESSED_DATA_DIR / 'vocab.txt', 'r', encoding='utf-8') as f:
        vocab_list = [line.strip() for line in f.readlines()]
    model = InputMethodModel(vocab_size=len(vocab_list))
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt'))
    model.to(device)
    dataloader = get_dataloader(train=False)

    # 评估模型
    top1_acc, top5_acc = evaluate_model(model, dataloader, device)
    print("============ 评估结果 ============")
    print(f"top1_acc: {top1_acc:.4f}")
    print(f"top5_acc: {top5_acc:.4f}")


if __name__ == '__main__':
    run_evaluate()
