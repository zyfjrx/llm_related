import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import os

from transformers import AutoTokenizer, AutoProcessor

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class VLMDataset(Dataset):
    def __init__(self, jsonl_path, images_path, tokenizer, preprocess=None, max_length=512,
                 image_special_token='<|image_pad|>' * 196):

        super().__init__()
        self.samples = self.load_data(jsonl_path)
        self.images_path = images_path

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = preprocess
        self.image_token = image_special_token
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        messages = []
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content'].replace('<image>', self.image_token)})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image_paths = sample['image']
        prompt = self._create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        loss_mask = self._generate_loss_mask(input_ids)

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)

        image_tensors = []
        for image_name in image_paths.split(','):
            try:
                image_name = image_name.strip()
                image = Image.open(f'{self.images_path}/{image_name}')
            except:
                image = Image.new('RGB', (224, 224),color='white')
            image_tensor = self.preprocess(images=image, return_tensors='pt')['pixel_values']
            image_tensors.append(image_tensor)
        image_tensors = torch.stack(image_tensors, dim=0)

        return {
            'input_ids': X,
            'labels': Y,
            'attention_mask': loss_mask,
            'pixel_tensors': image_tensors
        }

if __name__ == '__main__':
    jsonl_path = "/home/bmh/project/llm_related/train_multimodal/data/pretrain_vlm_data.jsonl"
    images_path = "/home/bmh/project/llm_related/train_multimodal/data/pretrain_images"
    tokenizer = AutoTokenizer.from_pretrained("/home/bmh/project/model/Qwen/Qwen2.5-0.5B-Instruct")
    processor = AutoProcessor.from_pretrained("/home/bmh/project/model/AI-ModelScope/siglip-base-patch16-224")
    ds = VLMDataset(jsonl_path, images_path, tokenizer, processor)
    print(len(ds))

    # print(tokenizer.decode(62182))
    # print(tokenizer.decode(151643))
    # print(tokenizer.encode("<|image_pad|>"))
    # print(tokenizer.pad_token_id)