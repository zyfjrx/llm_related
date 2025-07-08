import torch
from tqdm import tqdm
from enum import Enum
from conf import config
from preprocess.dataset import get_dataloader,get_dataset,DataType
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
from models.bert_classifier import ProductClassifyModel
from runner.predict import predict_batch


class Metric(Enum):
    ACCURACY = "accuracy" # 准确率
    PRECISION = "precision" # 精确率
    RECALL = "recall" # 召回率
    F1 = "f1" # F1分数
def evaluate(model, dataloader, device,metrics):
    model.eval()
    all_preds = []
    all_labels = []
    for batch in tqdm(dataloader, desc='evaluate'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['label'].tolist()
        predict_result = predict_batch(model, input_ids, attention_mask)

        all_labels.extend(targets)
        all_preds.extend(predict_result)
    results = {}
    for metric in metrics:
        if metric == Metric.ACCURACY:
            results[metric.value] = accuracy_score(all_labels, all_preds)
        elif metric == Metric.PRECISION:
            results[metric.value] = precision_score(all_labels, all_preds, average='macro')
        elif metric == Metric.RECALL:
            results[metric.value] = recall_score(all_labels, all_preds, average='macro')
        elif metric == Metric.F1:
            results[metric.value] = f1_score(all_labels, all_preds, average='macro')
    return results


def run_evaluate():
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"设备: {device}")
    # 加载词表
    model = ProductClassifyModel()
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pt'))
    model.to(device)

    dataloader = get_dataloader(DataType.TEST)

    # 指定评估的指标
    metrics = [Metric.ACCURACY, Metric.PRECISION, Metric.RECALL, Metric.F1]

    results = evaluate(model, dataloader, device,metrics)
    print("========== 评估结果 ==========")
    for name, value in results.items():
        print(f"{name}: {value:.4f}")
    print("=============================")

if __name__ == '__main__':
    run_evaluate()