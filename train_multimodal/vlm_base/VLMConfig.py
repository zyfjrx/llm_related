from transformers import PretrainedConfig
from typing import List



class VLMConfig(PretrainedConfig):
    model_type = 'vlm_model'
    def __init__(self,
                 image_special_token: str = '<|image_pad|>' * 196,
                 image_ids: List = [151655] * 196,
                 llm_model_path = "/home/bmh/project/model/Qwen/Qwen2.5-0.5B-Instruct",
                 vision_model_path = "/home/bmh/project/model/clip-vit-base-patch16",
                   **kwargs):
        self.image_special_token = image_special_token
        self.image_ids = image_ids
        self.vision_model_path = vision_model_path
        self.llm_model_path = llm_model_path
        super().__init__(**kwargs)