from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoConfig
from PIL import Image
from vlm_base_siglip.config import VLMConfig
from train_multimodal.vlm_base_clip.model import VLM
from transformers import CLIPProcessor,CLIPModel,AutoModelForCausalLM
import torch

device = "mps"
tokenizer = AutoTokenizer.from_pretrained("/home/bmh/project/model/Qwen/Qwen2.5-0.5B-Instruct")
processor = CLIPProcessor.from_pretrained("/home/bmh/project/model/clip-vit-base-patch16")
AutoConfig.register("vlm_model", VLMConfig)
AutoModelForCausalLM.register(VLMConfig, VLM)

model = AutoModelForCausalLM.from_pretrained('/home/bmh/project/llm_related/train_multimodal/vlm_base/save/pretrain/checkpoint-2400')

print(model)

for name, param in model.named_parameters():
    if 'linear' in name or 'vision_model':
        param.requires_grad = False
    if 'llm_model' in name:
        param.requires_grad = True
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
print(f'模型参数量为：{sum(p.numel() for p in model.parameters())}')
print(f'模型可训练参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
