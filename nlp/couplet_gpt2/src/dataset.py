
from torch.utils.data import DataLoader,Dataset
import config
from datasets import load_dataset,load_from_disk



def get_dataloader(train=True):
    data_path = str(config.PROCESSED_DATA_DIR / ('train.jsonl' if train else 'test.jsonl'))
    dataset = load_dataset('json',data_files=data_path)['train']
    dataset.set_format('torch')
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=train,drop_last=True)

if __name__ == '__main__':
    dataloader = get_dataloader()
    for batch in dataloader:
        print(batch['input_ids'].shape)  # [bs,l],[bs]
        print(batch['attention_mask'].shape)
        break