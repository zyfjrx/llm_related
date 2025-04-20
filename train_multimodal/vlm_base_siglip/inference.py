from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoConfig
from PIL import Image

from train_multimodal.vlm_base_siglip.model import VLM
from config import VLMConfig
import torch
from torch.nn import functional as F


def init_model():
    device = "mps"
    tokenizer = AutoTokenizer.from_pretrained("/Users/zhangyf/llm/Qwen2.5-0.5B-Instruct")
    processor = AutoProcessor.from_pretrained("/Users/zhangyf/llm/siglip-base-patch16-224")
    AutoConfig.register("vlm_model", VLMConfig)
    AutoModelForCausalLM.register(VLMConfig, VLM)

    pretrain_model = AutoModelForCausalLM.from_pretrained(
        '/Users/zhangyf/PycharmProjects/train/llm_related/train_multimodal/vlm_base_siglip/save')
    pretrain_model.to(device)
    pretrain_model.eval()
    return pretrain_model, processor, tokenizer


def generate(model, tokenizer, processor, image_input, text_input, max_new_tokens=200, temperature=0.0, top_k=None,
             device="mps"):
    q_text = tokenizer.apply_chat_template([{"role": "system", "content": 'You are a helpful assistant.'},
                                            {"role": "user", "content": f'{text_input}\n<image>'}], \
                                           tokenize=False, \
                                           add_generation_prompt=True).replace('<image>', '<|image_pad|>' * 49)
    input_ids = tokenizer(q_text, return_tensors='pt')['input_ids']
    input_ids = input_ids.to(device)
    pixel_values = processor(images=image_input).pixel_values
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
    model, processor, tokenizer = init_model()
    print(model)

    image_input = "/Users/zhangyf/Documents/WechatIMG443.jpg"
    image = Image.open(image_input).convert("RGB")
    text = "描述下这个图片"
    out = generate(model, tokenizer, processor, image, text)
    print(out)
