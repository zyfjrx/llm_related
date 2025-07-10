import torch
from transformers import AutoTokenizer,BertModel

def load_params(model,model_params_path):
    try:
        model.load_state_dict(torch.load(model_params_path))
    except (FileNotFoundError,AttributeError):
        print("模型参数文件不存在，使用默认参数")
    except RuntimeError:
        model.load_state_dict(torch.load(model_params_path,map_location=torch.device('cpu')))


class AddressTagging(torch.nn.Module):
    def __init__(self, model_name, label_list):
        super().__init__()
        self.label_list = label_list
        self.num_labels = len(label_list)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, self.num_labels)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(self.dropout(outputs.last_hidden_state))
        loss = 0.0
        if labels is not None:
            loss += self.loss_fn(logits.view(-1,self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits}