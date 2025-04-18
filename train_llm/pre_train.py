from model.config import LMConfig
from model.model import MiniMindLM
from transformers import DefaultDataCollator, AutoTokenizer, TrainingArguments, Trainer
import warnings
from model.dataset import PretrainDataset
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    config = LMConfig(use_moe=True,dim=640)
    model = MiniMindLM(config)
    print(model)
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    data_collator = DefaultDataCollator()
    tokenizer = AutoTokenizer.from_pretrained("model/tokenizer", use_fast=True)
    output_dir = 'model/save/pretrain/moe_model'
    args = TrainingArguments(output_dir=output_dir,
                             num_train_epochs=10,
                             do_train=True,
                             per_device_train_batch_size=32,
                             gradient_accumulation_steps=8,
                             logging_steps=100,
                             save_steps=100,
                             save_total_limit=2,
                             bf16=True,
                             learning_rate=5e-4,
                             lr_scheduler_type='cosine',
                             dataloader_num_workers=8,
                             dataloader_pin_memory=True,
                             save_safetensors=False
                             )
    dataset = PretrainDataset('../data/pretrain_hq.jsonl', tokenizer=tokenizer, max_length=512)
    trainer = Trainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer, data_collator=data_collator)
    # 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(output_dir)
    trainer.save_state()