from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoConfig
from PIL import Image
from VLMConfig import VLMConfig
from train_multimodal.vlm_base.model import VLM
from transformers import CLIPProcessor,CLIPModel,AutoModelForCausalLM
import torch

device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("/home/bmh/project/model/Qwen/Qwen2.5-0.5B-Instruct")
processor = CLIPProcessor.from_pretrained("/home/bmh/project/model/clip-vit-base-patch16")
model = CLIPModel.from_pretrained("/home/bmh/project/model/clip-vit-base-patch16")
model2 = AutoModelForCausalLM.from_pretrained("/home/bmh/project/model/Qwen/Qwen2.5-0.5B-Instruct")



print(model.config.vision_config.hidden_size)
print(model2.config.hidden_size)
