import config
from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer

def process():
    print("开始处理数据")
    # 读取数据
    dataset = load_dataset('csv', data_files=str(config.RAW_DATA_DIR / 'online_shopping_10_cats.csv'))['train']
    # 过滤数据
    dataset = dataset.remove_columns(['cat'])
    dataset = dataset.filter(lambda x: x['review'] is not None)
    print(dataset.features)
    # 划分数据集
    dataset = dataset.cast_column('label', ClassLabel(num_classes=2))
    print(dataset.features)
    dataset_dict = dataset.train_test_split(test_size=0.2, stratify_by_column='label')
    print(dataset_dict)


    # 构建tokenizer
    tokenizer = AutoTokenizer.from_pretrained('/Users/zhangyf/llm/bert-base-chinese')

    # 构建训练数据
    def batch_fn(batch):
        return tokenizer(batch['review'], truncation=True, padding='max_length', max_length=config.SEQ_LEN)
    dataset = dataset_dict.map(batch_fn, batched=True,remove_columns=['review'])
    dataset = dataset.remove_columns('token_type_ids')
    dataset['train'].save_to_disk(config.PROCESSED_DATA_DIR / 'train')
    dataset['test'].save_to_disk(config.PROCESSED_DATA_DIR / 'test')


    print("数据处理完成")

if __name__ == '__main__':
    process()