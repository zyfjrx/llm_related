from torch.utils.data import DataLoader,Dataset
import re
class MyDataset(Dataset):
    def __init__(self):
        self.lines = []
        with open('data/chinese_poems.txt',encoding="utf-8") as f:
            symbols = "，。、？！：.,()《》[]「」{}"
            pattern = "[" + re.escape(symbols) + "]"
            for line in f:
                line = re.sub(pattern, "", line).strip()
                self.lines.append(line)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        return self.lines[index]


dataset = MyDataset()
print(len(dataset))