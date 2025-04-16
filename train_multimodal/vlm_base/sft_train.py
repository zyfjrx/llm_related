import warnings
from transformers import CLIPProcessor, CLIPModel, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, \
    TrainingArguments, Trainer, DefaultDataCollator
from VLMConfig import VLMConfig
from dataset import PreDataset
from model import VLM
import os
import swanlab

key = os.getenv("SWANLAB_KEY")
swanlab.login(api_key=key, save=True)
swanlab.init(
  # 设置将记录此次运行的项目信息
  project="train_vlm",
  workspace="aigc_zyf",
  experiment_name="sft_train_vlm",
)
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    config = VLMConfig()
    model = VLM(config).cuda()
    print(model)
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    images_path = '/home/bmh/project/llm_related/train_multimodal/data/pretrain_images'
    data_path = '/home/bmh/project/llm_related/train_multimodal/data/pretrain_vlm_data.jsonl'
    tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)
    processor = CLIPProcessor.from_pretrained(config.vision_model_path)
    data_collator = DefaultDataCollator()
    output_dir = 'save/pretrain'
    args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        per_device_train_batch_size=2,
        learning_rate=1e-4,
        num_train_epochs=5,
        save_steps=100,
        save_total_limit=2,
        fp16=True,
        gradient_accumulation_steps=8,
        logging_steps=100,
        report_to='swanlab',
        dataloader_pin_memory=True,
        dataloader_num_workers=8
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=PreDataset(jsonl_path=data_path, images_path=images_path, tokenizer=tokenizer,
                                 preprocess=processor),
        data_collator=data_collator,
    )
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('save/pretrain')
    trainer.save_state()
