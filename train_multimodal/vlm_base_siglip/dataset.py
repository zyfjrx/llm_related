import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import os
from transformers import AutoTokenizer, AutoProcessor

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SFTDataset(Dataset):
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

    def _generate_labels_mask(self, input_ids):
        loss_mask = [self.tokenizer.pad_token_id] * len(input_ids)
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
                    loss_mask[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index: int):
        try:
            sample = self.samples[index]
            image_name = sample['image']
            image_name = image_name.strip()
            image = Image.open(f'{self.images_path}/{image_name}')
            prompt = self._create_chat_prompt(sample['conversations'])
            input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
            input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
            Y = self._generate_labels_mask(input_ids)[1:]
            X = torch.tensor(input_ids[:-1], dtype=torch.long)
            pixel_tensors = self.preprocess(images=image)['pixel_values']
        except:

            messages = [
                {"role": "user", "content": "图片内容是什么\n<image>"},
                {"role": "assistant", "content": '图片内容为空！'},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            ).replace('<image>', self.image_token)
            input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
            input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
            Y = self._generate_labels_mask(input_ids)[1:]
            X = torch.tensor(input_ids[:-1], dtype=torch.long)
            image = Image.new('RGB', (224, 224), color='white')
            pixel_tensors = self.preprocess(images=image)['pixel_values']
        return {
            'input_ids': X,
            'labels': Y,
            'pixel_values': pixel_tensors
        }


class PreDataset(Dataset):
    def __init__(self, jsonl_path, images_path, tokenizer, preprocess=None, max_length=512,
                 image_special_token='<|image_pad|>' * 49):

        super().__init__()
        self.samples = self.load_data(jsonl_path)
        self.images_path = images_path

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = preprocess
        self.image_token = image_special_token

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

    def __getitem__(self, index: int):
        try:

            sample = self.samples[index]
            image_name = sample['image']
            image_name = image_name.strip()
            image = Image.open(f'{self.images_path}/{image_name}')
            prompt = self._create_chat_prompt(sample['conversations'])
            input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
            input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
            X = torch.tensor(input_ids[:-1], dtype=torch.long)
            Y = torch.tensor(input_ids[1:], dtype=torch.long)

            pixel_tensors = self.preprocess(images=image)['pixel_values']
        except:

            messages = [
                {"role": "user", "content": "图片内容是什么\n<image>"},
                {"role": "assistant", "content": '图片内容为空！'},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            ).replace('<image>', self.image_token)
            input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
            input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
            X = torch.tensor(input_ids[:-1], dtype=torch.long)
            Y = torch.tensor(input_ids[1:], dtype=torch.long)
            image = Image.new('RGB', (224, 224), color='white')
            pixel_tensors = self.preprocess(images=image)['pixel_values']
        return {
            'input_ids': X,
            'labels': Y,
            'pixel_values': pixel_tensors
        }

# if __name__ == '__main__':
#     tokenizer = AutoTokenizer.from_pretrained("/home/bmh/project/model/Qwen/Qwen2.5-0.5B-Instruct")
#     processor = AutoProcessor.from_pretrained("/home/bmh/project/model/clip-vit-base-patch16")
#     jsonl_path = "/home/bmh/project/llm_related/train_multimodal/data/sft_test_data.jsonl"
#     image_input = "/home/bmh/project/llm_related/train_multimodal/data/test_image"
#     ds = SFTDataset(jsonl_path, image_input, tokenizer,preprocess=processor)
#     print(tokenizer.decode(ds[0]["input_ids"]))
#     print(tokenizer.decode(ds[0]["labels"]))