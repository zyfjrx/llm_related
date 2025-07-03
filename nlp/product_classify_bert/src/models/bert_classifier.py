from torch import nn
from transformers import BertModel,AutoTokenizer
from conf import config

class ProductClassifyModel(nn.Module):
    def __init__(self,freeze_bret=True):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.PRETRAINED_MODELS_DIR)
        self.linear = nn.Linear(self.bert.config.hidden_size,config.NUM_CLASS)
        if freeze_bret:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self,input_ids,attention_mask):
        output = self.bert(input_ids=input_ids,attention_mask=attention_mask)
        # last_hidden.shape[batch_size,seq_len,hidden_size]
        last_hidden = output.last_hidden_state
        # last_hidden.shape[batch_size,hidden_size]
        last_hidden = last_hidden[:,0,:]
        return self.linear(last_hidden)