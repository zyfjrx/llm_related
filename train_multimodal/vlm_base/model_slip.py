from torch import nn
import torch.nn.functional as F
import warnings
from transformers import AutoProcessor, AutoModel, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, \
    TrainingArguments, Trainer, DefaultDataCollator
import torch
from transformers.modeling_outputs import CausalLMOutputWithPast
from VLMConfig import VLMConfig
import swanlab
import os

warnings.filterwarnings('ignore')
swanlab.login(os.getenv("SWANLAB_KEY"), save=True)


class VisionProj(nn.Module):
    def __init__(self, ve_dim, lm_dim):
        super().__init__()
        self.ve_dim = ve_dim
        self.lm_dim = lm_dim
        self.linear1 = nn.Linear(self.ve_dim * 4, self.lm_dim)
        self.linear2 = nn.Linear(self.lm_dim, self.lm_dim)

    def forward(self, image_encoders):
        vision_proj = self.linear2(F.silu(self.linear1(image_encoders)))
        return vision_proj


class VLM(PreTrainedModel):
    config_class = VLMConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vision_model = AutoModel.from_pretrained(self.config.vision_model_path)
        self.processor = AutoProcessor.from_pretrained(self.config.vision_model_path)
        self.llm_model = AutoModelForCausalLM.from_pretrained(self.config.llm_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_path)
        self.visionProj = VisionProj(lm_dim=self.llm_model.config.hidden_size,
                                     ve_dim=self.vision_model.config.vision_config.hidden_size * 4)
        if self.config.freeze_vision_model:
            for param in self.vision_model.parameters():
                param.requires_grad = False
        for param in self.llm_model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, labels, pixel_values, attention_mask=None):
        text_embeds = self.llm_model.get_input_embeddings()(input_ids)
        pixel_values = pixel_values.squeeze(1)
        image_embeds = self.vision_model.vision_model(pixel_values).last_hidden_state[:, 1:, :]
        b, s, d = image_embeds.shape
        image_embeds = image_embeds.view(b, -1, d * 4)  # (b, 196, d) --> (b, 49, d*4) 压缩图片tokens
        image_features = self.visionProj(image_embeds)
        text_embeds = text_embeds.to(image_features.dtype)
        inputs_embeds = self.merge_input_ids_with_image_features(image_features, text_embeds, input_ids)
        outputs = self.llm_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs[0]
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = loss_fct(
                logits.view(-1, logits.size(-1)), labels.view(-1).to(logits.device)
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids):
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_indices, image_indices = torch.where(input_ids == self.tokenizer('<|image_pad|>')['input_ids'][0])
        inputs_embeds[batch_indices, image_indices] = image_features.view(-1, embed_dim)
        return inputs_embeds
