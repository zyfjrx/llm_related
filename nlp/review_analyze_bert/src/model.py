import torch.nn as nn
import config
import torch
from transformers import AutoModel
class ReviewAnalyzeModel(nn.Module):
    def __init__(self,freeze_bert=True):
        super(ReviewAnalyzeModel, self).__init__()
        self.bert = AutoModel.from_pretrained(config.PRETRAINED_MODELS_DIR)
        self.linear = nn.Linear(self.bert.config.hidden_size,1)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self,input_ids,attention_mask):
        # input_ids.shape [batch_size, seq_len]
        # pooler_output.shape [batch_size, hidden_size]
        pooler_output = self.bert(input_ids,attention_mask)[1]
        out = self.linear(pooler_output) # out.shape [batch_size, 1]
        return out.squeeze(1) # out.shape [batch_size]


if __name__ == '__main__':
    model = ReviewAnalyzeModel()
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')