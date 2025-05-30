from train_llm.v1.model.config import LMConfig
from train_llm.v1.model.model import MiniMindLM
from transformers import DefaultDataCollator, AutoTokenizer, TrainingArguments, Trainer
import warnings
import torch
import swanlab
import os
from train_llm.v1.model.dataset import SFTDataset
warnings.filterwarnings('ignore')
key = os.getenv("SWANLAB_KEY")
swanlab.login(api_key=key, save=True)
swanlab.init(
  # 设置将记录此次运行的项目信息
  project="train_llm",
  workspace="aigc_zyf",
  experiment_name="sft_train_llm",
)

if __name__ == '__main__':
    config = LMConfig(use_moe=False)
    tokenizer = AutoTokenizer.from_pretrained("model/tokenizer", use_fast=True)
    model = MiniMindLM(config)
    state_dict = torch.load("/home/bmh/project/llm_related/train_llm/model/save/pretrain/minimind_base_pre/pytorch_model.bin",map_location='cuda')
    model.load_state_dict(state_dict,strict=False)
    print(model)
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    data_collator = DefaultDataCollator()
    output_dir = 'v1/model/save/sft/full-sft/epoch1'
    args = TrainingArguments(output_dir=output_dir,
                             num_train_epochs=1,
                             do_train=True,
                             per_device_train_batch_size=64,
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
    dataset = SFTDataset('/home/bmh/project/llm_related/train_llm/data/sft/sft_mini_512 .jsonl', tokenizer=tokenizer, max_length=512)
    trainer = Trainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer, data_collator=data_collator)
    # 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(output_dir)
    trainer.save_state()