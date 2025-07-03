import pandas as pd
import config
from transformers import AutoTokenizer
from datasets import load_dataset

def process():
    print("开始处理数据")
    # 读取数据
    datasets = load_dataset('json', data_files=str(config.RAW_DATA_DIR / 'couplet_combined.jsonl'))['train']
    # 划分训练测试数据集
    dataset_dict = datasets.train_test_split(test_size=0.2)

    # 获取tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_MODELS_DIR)
    # 构建并保存训练数据集
    def encode_fn(example):
        return tokenizer(example["couplet"], truncation=True, padding="max_length", max_length=config.SEQ_LEN)
    dataset_dict = dataset_dict.map(encode_fn, batched=True, remove_columns=['couplet'])
    dataset_dict['train'].to_json(config.PROCESSED_DATA_DIR / 'train.jsonl')
    dataset_dict['test'].to_json(config.PROCESSED_DATA_DIR / 'test.jsonl')



    # 构建并保存测试数据集

    print("数据处理完成")


if __name__ == '__main__':
    process()
