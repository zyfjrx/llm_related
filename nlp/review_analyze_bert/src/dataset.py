from torch.utils.data import DataLoader
import config
from datasets import load_from_disk


def get_dataloader(train=True):
    data_path = str(config.PROCESSED_DATA_DIR / ('train' if train else 'test'))
    dataset = load_from_disk(data_path)
    dataset.set_format('torch')
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=train)

if __name__ == '__main__':
    dataloader = get_dataloader(train=True)
    print(len(dataloader))
    for batch in dataloader:
        print(batch['input_ids'].shape, batch['label'].shape)  #[bs,l],[bs]
        print(batch['attention_mask'].shape)
        break