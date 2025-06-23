import re
import torch
from torch import nn,optim
from torch.utils.data import DataLoader,Dataset


def process_poems(file_path):
    poems = []
    char_set = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = re.sub(r"[，。、？！：]", "", line).strip()
            char_set.update(list(line))
            poems.append(list(line))

    vocab = list(char_set) + ["<UNK>"]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    sequences = []
    for poem in poems:
        seq = [word2idx.get(word) for word in poem]
        sequences.append(seq)
    return sequences,word2idx,vocab
sequences,word2idx,vocab = process_poems('../data/poems.txt')
print(len(sequences))
print(len(word2idx))
print(len(vocab))

class PoemDataset(Dataset):
    def __init__(self, sequences, seq_len):
        self.seq_len = seq_len
        self.data = []
        for seq in sequences:
            for i in range(0, len(seq) - self.seq_len):
                self.data.append((seq[i:i+self.seq_len], seq[i+1:i+1+self.seq_len]))

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x = torch.LongTensor(self.data[idx][0])
        y = torch.LongTensor(self.data[idx][1])
        return x, y
dataset = PoemDataset(sequences,20)

# 搭建模型
class PoemRNN(nn.Module):
    def __init__(self, vocab_size,embedding_dim=128, hidden_size=256, num_layers=1):
        super(PoemRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        output, hidden = self.rnn(x)
        output = self.linear(output)
        return output, hidden

device = torch.device('mps' if torch.mps.is_available() else 'cpu')
model = PoemRNN(vocab_size=len(vocab), embedding_dim=128, hidden_size=256, num_layers=2).to(device)

def train(model,dataset,lr,epochs,batch_size,device):
    model.train()
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loss_list = []
    for epoch in range(epochs):
        train_total_loss = 0
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            output,hx = model(X)
            # output = output.view(-1, output.shape[-1])
            loss = loss_func(output.view(-1, output.shape[-1]), y.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_total_loss += loss.item() * X.shape[0]
        train_avg_loss = train_total_loss / len(dataset)
        train_loss_list.append(train_avg_loss)
        print(f"epoch: {epoch + 1},train loss: {train_avg_loss:.6f}")
    return train_loss_list

train(model=model, dataset=dataset, lr=1e-3, epochs=20, batch_size=32, device=device)
torch.save(model.state_dict(), 'poems.pth')

def generate(model,word2idx,vocab,start_token,line_num=4,line_len=7):
    model.eval()
    poem = []
    current_line_len = line_len
    start_token = word2idx.get(start_token,word2idx["<UNK>"])
    if start_token != word2idx["<UNK>"]:
        poem.append(vocab[start_token])
        current_line_len -= 1
    # 神经网络输入 [1,1] ,b,L
    input = torch.LongTensor([[start_token]]).to(device)
    with torch.no_grad():
        for _ in range(line_num):
            for interpunction in ["，","。\n"]:
                while current_line_len > 0:
                    output,_ = model(input)
                    prob = torch.softmax(output[:,-1,:],dim=-1)
                    next_token = torch.multinomial(prob,1)
                    poem.append(vocab[next_token.item()])
                    input = torch.cat((input,next_token),dim=1)
                    current_line_len -= 1
                current_line_len = line_len
                poem.append(interpunction)

        return "".join(poem)
print(generate(model,word2idx,vocab,start_token="山",line_num=4,line_len=7))

