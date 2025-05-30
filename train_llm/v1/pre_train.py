import os
import swanlab
from train_llm.v1.model.config import LMConfig
from train_llm.v1.model.model import MiniMindLM
from transformers import DefaultDataCollator, AutoTokenizer, TrainingArguments, Trainer
import warnings
from train_llm.v1.model.dataset import PretrainDataset
warnings.filterwarnings('ignore')
key = os.getenv("SWANLAB_KEY")
swanlab.login(api_key=key, save=True)
swanlab.init(
  # 设置将记录此次运行的项目信息
  project="train_llm",
  workspace="aigc_zyf",
  experiment_name="pre_train_llm",
)


if __name__ == '__main__':
    config = LMConfig(use_moe=True,dim=640)
    model = MiniMindLM(config)
    print(model)
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    data_collator = DefaultDataCollator()
    tokenizer = AutoTokenizer.from_pretrained("model/tokenizer", use_fast=True)
    output_dir = 'v1/model/save/pretrain/moe_epoch10'
    args = TrainingArguments(output_dir=output_dir,
                             num_train_epochs=10,
                             do_train=True,
                             per_device_train_batch_size=16,
                             gradient_accumulation_steps=8,
                             logging_steps=10,
                             save_steps=100,
                             save_total_limit=2,
                             bf16=True,
                             learning_rate=5e-4,
                             lr_scheduler_type='cosine',
                             dataloader_num_workers=8,
                             dataloader_pin_memory=True,
                             save_safetensors=False
                             )
    dataset = PretrainDataset('/home/bmh/project/llm_related/train_llm/data/pre/pretrain_hq.jsonl', tokenizer=tokenizer, max_length=512)
    trainer = Trainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer, data_collator=data_collator)
    # 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(output_dir)
    trainer.save_state()