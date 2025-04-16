from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoConfig
from PIL import Image
from model import VLMConfig, VLM
from transformers import CLIPProcessor
import torch
from torch.nn import functional as F


def init_model(conf: VLMConfig, model_path):
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained("/home/bmh/project/model/Qwen/Qwen2.5-0.5B-Instruct")
    processor = CLIPProcessor.from_pretrained("/home/bmh/project/model/clip-vit-base-patch16")
    AutoConfig.register("vlm_model", VLMConfig)
    AutoModelForCausalLM.register(VLMConfig, VLM)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model, processor, tokenizer


def generate(model, tokenizer, processor, image_input, text_input, max_new_tokens=200, temperature=0.0, top_k=None,
             device="cuda"):
    q_text = tokenizer.apply_chat_template([{"role": "system", "content": 'You are a helpful assistant.'},
                                            {"role": "user", "content": f'{text_input}\n<image>'}], \
                                           tokenize=False, \
                                           add_generation_prompt=True).replace('<image>', '<|image_pad|>' * 49)
    input_ids = tokenizer(q_text, return_tensors='pt')['input_ids']
    input_ids = input_ids.to(device)
    pixel_values = processor(text=None, images=image_input, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    eos = tokenizer.eos_token_id
    s = input_ids.shape[1]
    while input_ids.shape[1] < s + max_new_tokens - 1:
        inference_res = model(input_ids, None, pixel_values)
        logits = inference_res.logits
        logits = logits[:, -1, :]

        for token in set(input_ids.tolist()[0]):
            logits[:, token] /= 1.0

        if temperature == 0.0:
            _, idx_next = torch.topk(logits, k=1, dim=-1)
        else:
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1, generator=None)

        if idx_next == eos:
            break

        input_ids = torch.cat((input_ids, idx_next), dim=1)
    return tokenizer.decode(input_ids[:, s:][0])


if __name__ == '__main__':
    conf = VLMConfig()
    model_path = "/home/bmh/project/llm_related/train_multimodal/vlm_base/save/pretrain/checkpoint-2600"
    model, processor, tokenizer = init_model(conf, model_path)
    image_input = "/home/bmh/project/llm_related/train_multimodal/data/test_image/GCC_train_000190697.jpg"
    image = Image.open(image_input).convert("RGB")
    text = "描述下这个图片"
    out = generate(model, tokenizer, processor, image, text)
    print(out)
