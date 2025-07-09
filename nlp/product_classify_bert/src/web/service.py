import torch
from transformers import AutoTokenizer

from conf import config
from models.bert_classifier import ProductClassifyModel
from preprocess.dataset import get_dataset, DataType
from runner.predict import predict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ProductClassifyModel(freeze_bret=False).to(device)
model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pt'))
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_MODELS_DIR)
dataset = get_dataset(DataType.TRAIN)
class_label = dataset.features['label']
def predict_title(text):
    return predict(text, model, tokenizer, device,class_label)