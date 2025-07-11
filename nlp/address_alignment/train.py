from tqdm import tqdm
import torch
from torch import optim
from sklearn.metrics import precision_recall_fscore_support


class Trainer:
    """训练验证与测试"""

    def __init__(self, model, device, epochs, lr, checkpoint_step=200):
        """
        参数：
        :param model:
        :param device:
        :param epochs:
        :param lr:
        :param checkpoint_step:
        """
        self.model = model
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.checkpoint_step = checkpoint_step

    def __call__(
            self,
            dataloader: dict,
            model_params_path=None,
            writer=None,
            is_test=False,
    ):
        """
          参数:
          - dataloader: 数据加载器
          - model_params_path: 模型参数保存路径
          - writer: 记录器
          - is_test: 是否执行测试
        """
        self.dataloader = dataloader
        self.model_params_path = model_params_path
        self.writer = writer
        self.is_test = is_test
        self.model.to(self.device)
        self.global_step = 0

        if is_test:
            for k,v in self.run_epoch("test").items():
                print(f"Test {k}:", v)
            return
        assert self.model_params_path is not None,"缺少模型参数保存路径"
        best_vaild_metric = 0
        for epoch in range(self.epochs):
            print(f"Epoch {epoch}:")
            train_metrics = self.run_epoch("train", epoch)
            for k,v in train_metrics.items():
                print(f"Train {k}:", v)

            valid_metrics = self.run_epoch("valid", epoch)
            for k,v in valid_metrics.items():
                print(f"Valid {k}:", v)

            if valid_metrics['f1'] > best_vaild_metric:
                best_vaild_metric = valid_metrics['f1']
                print("保存模型参数...")
                torch.save(self.model.state_dict(), self.model_params_path)

    def run_epoch(self, phase, epoch=0):
        self.model.train() if phase == 'train' else self.model.eval()
        # 初始化总损失和总样本
        total_loss = 0.0
        total_examples = 0
        # 初始化记录
        records = {}
        with torch.set_grad_enabled(phase == 'train'):
            for inputs in tqdm(self.dataloader[phase], desc=phase):
                # 数据转移到设备
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs, loss = self.forward(inputs)
                if phase == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if self.writer:
                        self.writer.add_scalar('train_loss', loss.item(), self.global_step)
                    self.global_step += 1

                    # if (self.checkpoint_step
                    #         and self.global_step % self.checkpoint_step == 0
                    # ):
                    #     checkpoint_path = str(self.model_params_path) + ".checkpoint"
                    #     torch.save(self.model.state_dict(), checkpoint_path)
                current_batch_size = inputs["input_ids"].shape[0]
                total_loss += loss.item() * current_batch_size
                total_examples += current_batch_size
                if phase != 'train':
                    self.update_records(inputs, outputs, records)
        metrics = {"loss": total_loss / total_examples}
        if phase != 'train':
            self.compute_metrics(metrics,records)
            if self.writer is not None:
                for metric_name,value in metrics.items():
                    self.writer.add_scalar(f"{phase}/{metric_name}", value, epoch)
        return metrics

    def forward(self, inputs):
        """前向传播"""
        raise NotImplementedError

    def update_records(self, inputs, outputs, records):
        """更新记录"""
        raise NotImplementedError

    def compute_metrics(self, metrics, records):
        """计算评估指标"""
        raise NotImplementedError


class AddressTaggingTrainer(Trainer):
    def forward(self, inputs):
        outputs = self.model(**inputs)
        return outputs, outputs['loss']

    def update_records(self, inputs, outputs, records):
        """更新记录"""
        preds = outputs['logits'].argmax(dim=-1)
        labels = inputs['labels']
        mask = (inputs['attention_mask'] == 1) & (labels != -100)
        preds = preds[mask].view(-1).detach().cpu()
        labels = labels[mask].view(-1).detach().cpu()
        records.setdefault('preds', []).append(preds)
        records.setdefault('labels', []).append(labels)

    def compute_metrics(self, metrics, records):
        """计算评估指标"""
        all_labels = torch.cat(records['labels'], dim=0)
        all_preds = torch.cat(records['preds'], dim=0)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro',zero_division=0)
        metrics.update({"precision": precision, "recall": recall, "f1": f1})

