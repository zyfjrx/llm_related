import pandas as pd
import torch
from torch.utils.data import DataLoader,Dataset
import config

class InputMethodDataset(Dataset):
    def __init__(self,data_path):
        self.data = pd.read_json(data_path,orient='records',lines=True).to_dict(orient='records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_tensor = torch.tensor(self.data[idx]['input'], dtype=torch.long)
        target_tensor = torch.tensor(self.data[idx]['target'], dtype=torch.long)
        return input_tensor, target_tensor


def get_dataloader(train=True):
    data_path = config.PROCESSED_DATA_DIR / ('train.jsonl' if train else 'test.jsonl')
    dataset = InputMethodDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=train)
    return dataloader

if __name__ == '__main__':
    dataloader1 = get_dataloader()
    print(len(dataloader1))
    dataloader = get_dataloader(train=False)
    print(len(dataloader))
    for input_tensor, target_tensor in dataloader:
        print(input_tensor.shape, target_tensor.shape)  #[bs,l],[bs]
        break