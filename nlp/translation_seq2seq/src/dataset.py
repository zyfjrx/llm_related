import pandas as pd
import torch
from torch.utils.data import DataLoader,Dataset
import config

class TranslationDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_json(data_path,orient='records',lines=True).to_dict(orient='records')

    def __len__(self):
        return len(self.data)

    def __getitem__ (self, idx):
        inputs = torch.tensor(self.data[idx]['zh'],dtype=torch.long)
        targets = torch.tensor(self.data[idx]['en'],dtype=torch.long)
        return inputs,targets

def get_dataloader(train=True):
    data_path = config.PROCESSED_DATA_DIR / ('train.jsonl' if train else 'test.jsonl')
    dataset = TranslationDataset(data_path)
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=train,drop_last=True)

if __name__ == '__main__':
    dataloader = get_dataloader()
    for inputs,targets in dataloader:
        print(inputs.shape,targets.shape)
        break