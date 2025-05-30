from train_llm.v1.model.config import LMConfig
from train_llm.v1.model.model import MiniMindLM
from transformers import AutoTokenizer, TrainingArguments, Trainer,AutoConfig,AutoModelForCausalLM
import warnings
import torch
from train_llm.v1.model.dataset import DPODataset,DPODataCollator
from torch.nn import functional as F
import os
import swanlab

warnings.filterwarnings('ignore')
key = os.getenv("SWANLAB_KEY")
swanlab.login(api_key=key, save=True)
swanlab.init(
  # 设置将记录此次运行的项目信息
  project="train_llm",
  workspace="aigc_zyf",
  experiment_name="dpo_train_llm",
)

def logits_to_probs(logits, labels):
    # 计算log概率值，并取出labels对应的log概率值
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return probs


def mask_logits(logits, labels):
    # 过滤掉mask的log概率值
    new_logits = []
    for logit, label in zip(logits, labels):
        new_logits.append(logit[label != 0].sum().unsqueeze(0))
    return new_logits


def dpo_loss(ref_probs, probs, beta):
    # 将chosen和rejected分开
    def split_probs(probs):
        len_chosen = int(len(ref_probs) // 2)
        chosen_probs = probs[:len_chosen]
        reject_probs = probs[len_chosen:]
        return torch.cat(chosen_probs),torch.cat(reject_probs)

    chosen_ref_probs,reject_ref_probs = split_probs(ref_probs)
    chosen_probs ,reject_probs = split_probs(probs)

    pi_logratios = chosen_probs - reject_probs
    ref_logratios = chosen_ref_probs - reject_ref_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(logits * beta)
    return loss.mean()


class DPOTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        with torch.no_grad():
            ref_logits = ref_model(input_ids=input_ids, labels=labels).logits
        ref_probs = logits_to_probs(ref_logits, labels)
        ref_probs = mask_logits(ref_probs, labels)
        logits = model(input_ids=input_ids, labels=labels).logits
        probs = logits_to_probs(logits, labels)
        probs = mask_logits(probs, labels)
        loss = dpo_loss(ref_probs=ref_probs, probs=probs, beta=0.1)
        return loss


if __name__ == '__main__':
    AutoConfig.register("minimind",LMConfig)
    AutoModelForCausalLM.register(LMConfig,MiniMindLM)
    model = AutoModelForCausalLM.from_pretrained("/home/bmh/project/llm_related/train_llm/model/save/sft/full-sft/epoch1")
    print(model)
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    ref_model = AutoModelForCausalLM.from_pretrained("/home/bmh/project/llm_related/train_llm/model/save/sft/full-sft/epoch1").eval().to('cuda')


    tokenizer = AutoTokenizer.from_pretrained("/home/bmh/project/llm_related/train_llm/model/tokenizer", use_fast=True)
    output_dir = 'v1/model/save/dpo/dpo_model'
    args = TrainingArguments(output_dir=output_dir,
                             num_train_epochs=1,
                             do_train=True,
                             per_device_train_batch_size=16,
                             gradient_accumulation_steps=8,
                             logging_steps=10,
                             save_steps=100,
                             save_total_limit=2,
                             bf16=True,
                             learning_rate=1e-6,
                             lr_scheduler_type='cosine',
                             dataloader_num_workers=8,
                             dataloader_pin_memory=True,
                             save_safetensors=False
                             )
    dataset = DPODataset('/home/bmh/project/llm_related/train_llm/data/dpo/dpo.jsonl', tokenizer=tokenizer, max_length=512)
    trainer = DPOTrainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer, data_collator=DPODataCollator(tokenizer=tokenizer))
    # 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(output_dir)
    trainer.save_state()
