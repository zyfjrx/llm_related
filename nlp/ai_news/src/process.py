import config
from transformers import AutoTokenizer
from datasets import load_dataset,ClassLabel

def process():
    print("开始处理数据")
    # 读取数据
    category_dataset = load_dataset('json', data_files=str(config.RAW_DATA_DIR / 'category.jsonl'))['train']
    summary_dataset = load_dataset('json', data_files=str(config.RAW_DATA_DIR / 'summary.jsonl'))['train']

    all_labels = category_dataset.unique('category')
    category_dataset = category_dataset.cast_column('category',ClassLabel(names=all_labels))

    # 获取tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_MODELS_DIR)
    # 统计长度
    # df = summary_dataset.to_pandas()
    # text_len = df['text'].apply(lambda x: len(tokenizer.tokenize(x))).max()
    # summary_len = df['summary'].apply(lambda x: len(tokenizer.tokenize(x))).max()
    # print("text最大长度：",text_len)
    # print("text最大长度：",summary_len)
    # 构建并保存训练数据集
    def tokenize(batch):
        tokenized = tokenizer(batch['text'], truncation=True, padding='max_length', max_length=config.TEXT_SEQ_LEN)
        batch['input_ids'] = tokenized['input_ids']
        batch['attention_mask'] = tokenized['attention_mask']
        return  batch
    category_dataset = category_dataset.map(tokenize, batched=True,remove_columns=['text'])
    category_dataset.save_to_disk(config.PROCESSED_DATA_DIR / 'category')


    # 构建并保存测试数据集

    print("数据处理完成")


if __name__ == '__main__':
    process()
