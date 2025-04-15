from torch import nn
import warnings
from transformers import CLIPProcessor, CLIPModel, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, \
    TrainingArguments, Trainer, DefaultDataCollator
import torch
from transformers.modeling_outputs import CausalLMOutputWithPast
from VLMConfig import VLMConfig
from dataset import VLMDataset
import swanlab
import os
warnings.filterwarnings('ignore')
swanlab.login(os.getenv("SWANLAB_KEY"), save=True)

class VisionProj(nn.Module):
    def __init__(self, ve_dim=768, lm_dim=896):
        super().__init__()
        self.ve_dim = ve_dim
        self.lm_dim = lm_dim
        self.vision_proj = nn.Sequential(
            nn.Linear(self.ve_dim, self.lm_dim)
        )

    def forward(self, image_encoders):
        vision_proj = self.vision_proj(image_encoders)
        return vision_proj


class VLM(PreTrainedModel):
    config_class = VLMConfig

    def __init__(self, params: VLMConfig = None):
        super().__init__(params)
        if not params: params = VLMConfig()
        self.params = params
        self.vision_model = CLIPModel.from_pretrained(params.vision_model_path)
        self.processor = CLIPProcessor.from_pretrained(params.vision_model_path)
        self.vision_proj = VisionProj()
        self.llm_model = AutoModelForCausalLM.from_pretrained(params.llm_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(params.llm_model_path)
        for param in self.vision_model.parameters():
            param.requires_grad = False
        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False

    @staticmethod
    def get_vision_model(model_path="./model/vision_model/clip-vit-base-patch16"):
        model = CLIPModel.from_pretrained(model_path)
        processor = CLIPProcessor.from_pretrained(model_path)
        # 冻结 vision_encoder 的所有参数
        for param in model.parameters():
            param.requires_grad = False
        return model.eval(), processor

    @staticmethod
    def image2tensor(image, processor):
        if image.mode in ['RGBA', 'LA']: image = image.convert('RGB')
        inputs = processor(images=image, return_tensors="pt")['pixel_values']
        return inputs

    @staticmethod
    def get_image_embeddings(image_tensors, vision_model):
        with torch.no_grad():
            outputs = vision_model.vision_model(pixel_values=image_tensors)
        img_embedding = outputs.last_hidden_state[:, 1:, :].squeeze()
        return img_embedding

    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512):
        def find_indices(tokens, image_ids):
            image_ids_tensor = torch.tensor(image_ids).to(tokens.device)
            len_image_ids = len(image_ids)
            if len_image_ids > tokens.size(1):
                return None
            tokens_view = tokens.unfold(1, len_image_ids, 1)
            matches = (tokens_view == image_ids_tensor).all(dim=2)
            return {
                batch_idx: [(idx.item(), idx.item() + len_image_ids - 1) for idx in
                            matches[batch_idx].nonzero(as_tuple=True)[0]]
                for batch_idx in range(tokens.size(0)) if matches[batch_idx].any()
            } or None

        image_indices = find_indices(tokens, self.params.image_ids)
        if vision_tensors is not None and image_indices:
            vision_proj = self.vision_proj(vision_tensors)
            if len(vision_proj.shape) == 3:
                vision_proj = vision_proj.unsqueeze(0)
            new_h = []
            for i in range(h.size(0)):
                if i in image_indices:
                    h_i = h[i]
                    img_idx = 0
                    for start_idx, end_idx in image_indices[i]:
                        if img_idx < vision_proj.size(1):
                            h_i = torch.cat((h_i[:start_idx], vision_proj[i][img_idx], h_i[end_idx + 1:]), dim=0)[
                                  :seqlen]
                            img_idx += 1
                    new_h.append(h_i)
                else:
                    new_h.append(h[i])
            return torch.stack(new_h, dim=0)
        return h

    def forward(self, input_ids, labels, pixel_tensors, attention_mask=None):
        text_embeds = self.llm_model.get_input_embeddings()(input_ids)

        if pixel_tensors is not None:
            if len(pixel_tensors.shape) == 6:
                pixel_tensors = pixel_tensors.squeeze(2)
            bs, num, c, im_h, im_w = pixel_tensors.shape
            stack_dim = 1 if bs > 1 else 0
            vision_tensors = torch.stack([
                VLM.get_image_embeddings(pixel_tensors[:, i, :, :, :], self.vision_model)
                for i in range(num)
            ], dim=stack_dim)
            inputs_embeds = self.count_vision_proj(tokens=input_ids, h=text_embeds, vision_tensors=vision_tensors,
                                                   seqlen=input_ids.shape[1])

        outputs = self.llm_model(inputs_embeds=inputs_embeds,labels = labels, attention_mask=attention_mask)
        logits = outputs['logits']
        loss = outputs['loss']
        # loss = None
        # if labels is not None:
        #     loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        #     loss = loss_fct(
        #         logits.view(-1, logits.size(-1)), labels.view(-1).to(logits.device)
        #     )
        return CausalLMOutputWithPast(loss=loss, logits=logits)


