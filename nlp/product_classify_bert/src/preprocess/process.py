from datasets import load_dataset,ClassLabel
from conf import config
import datasets
from transformers import AutoTokenizer
import pandas as pd
def process_data():
    print("开始处理数据")
    # 读取数据
    dataset_dic = datasets.load_dataset('csv', data_files={
        'train': str(config.RAW_DATA_DIR / 'train.txt'),
        'test': str(config.RAW_DATA_DIR / 'test.txt'),
        'valid': str(config.RAW_DATA_DIR / 'valid.txt')
    }, delimiter='\t')

    # 过滤数据
    dataset_dic = dataset_dic.filter(lambda x: x['text_a'] is not None and x['label'] is not None)


    # 处理类别
    all_labels = dataset_dic['train'].unique('label')
    dataset_dic = dataset_dic.cast_column('label',ClassLabel(names=all_labels))

    # 获取tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_MODELS_DIR)

    # 统计长度
    # df = dataset_dic['train'].to_pandas()
    # zh_len = df['text_a'].apply(lambda x: len(tokenizer.tokenize(x))).max()
    # print("中文最大长度：",zh_len)

    def tokenize(batch):
        tokenized = tokenizer(batch['text_a'], truncation=True, padding='max_length', max_length=config.SEQ_LEN)
        batch['input_ids'] = tokenized['input_ids']
        batch['attention_mask'] = tokenized['attention_mask']
        return  batch
    dataset_dic = dataset_dic.map(tokenize, batched=True,remove_columns=['text_a'])
    # print(dataset_dic['train'][:3])
    # print(dataset_dic['train'].features['label'].int2str(5))
    # 保存数据
    dataset_dic['train'].save_to_disk(str(config.PROCESSED_DATA_DIR / 'train'))
    dataset_dic['test'].save_to_disk(str(config.PROCESSED_DATA_DIR / 'test'))
    dataset_dic['valid'].save_to_disk(str(config.PROCESSED_DATA_DIR / 'valid'))

    print("数据处理完成")



if __name__ == '__main__':
    process_data()