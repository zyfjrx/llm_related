from model.config import LMConfig
from model.model import MiniMindLM
from transformers import DefaultDataCollator, AutoTokenizer, TrainingArguments, Trainer,AutoConfig,AutoModelForCausalLM
import warnings
import torch
from model.dataset import DPODataset
from torch.nn import functional as F

warnings.filterwarnings('ignore')


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
    batch_size = ref_probs.size(0)
    chosen_ref_probs = ref_probs[:batch_size // 2]
    reject_ref_probs = ref_probs[batch_size // 2:]

    chosen_probs = probs[:batch_size // 2]
    reject_probs = probs[batch_size // 2:]

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
            ref_logits = model(input_ids=input_ids, labels=labels).logits
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
    model = AutoModelForCausalLM.from_pretrained("/home/bmh/project/llm_related/train_llm/model/save/pretrain/minimind_base_pre")
    print(model)
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    ref_model = AutoModelForCausalLM.from_pretrained("/home/bmh/project/llm_related/train_llm/model/save/full_sft/minimind_base_sft")
    data_collator = DefaultDataCollator()
    tokenizer = AutoTokenizer.from_pretrained("/home/bmh/project/llm_related/train_llm/model/tokenizer", use_fast=True)
    output_dir = 'model/save/pretrain/dpo_model'
    args = TrainingArguments(output_dir=output_dir,
                             num_train_epochs=1,
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
    dataset = DPODataset('../data/sft_mini_test.jsonl', tokenizer=tokenizer, max_length=512)
    trainer = DPOTrainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer, data_collator=data_collator)
    # 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(output_dir)
    trainer.save_state()
