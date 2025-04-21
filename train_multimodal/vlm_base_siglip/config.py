from transformers import PretrainedConfig
from typing import List



class VLMConfig(PretrainedConfig):
    model_type = 'vlm_model'
    def __init__(self,
                 image_special_token: str = '<|image_pad|>' * 49,
                 image_ids: List = [151655] * 49,
                 llm_model_path = "/Users/zhangyf/llm/Qwen2.5-0.5B-Instruct",
                 vision_model_path = "/Users/zhangyf/llm/siglip-base-patch16-224",
                 freeze_vision_model=True,
                   **kwargs):
        self.image_special_token = image_special_token
        self.image_ids = image_ids
        self.vision_model_path = vision_model_path
        self.llm_model_path = llm_model_path
        self.freeze_vision_model = freeze_vision_model
        super().__init__(**kwargs)