import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoConfig
import torch
from model.config import LMConfig
from model.model import MiniMindLM

device = "cuda"
AutoConfig.register("minimind", LMConfig)
AutoModelForCausalLM.register(LMConfig, MiniMindLM)
tokenizer = AutoTokenizer.from_pretrained("/home/bmh/project/llm_related/train_llm/model/save/pretrain/minimind_base_pre")
pretrain_model = AutoModelForCausalLM.from_pretrained('/home/bmh/project/llm_related/train_llm/model/save/pretrain/minimind_base_pre')
pretrain_model.to(device)

sft_model = AutoModelForCausalLM.from_pretrained("/home/bmh/project/llm_related/train_llm/model/save/sft/full-sft/epoch1")
sft_model.to(device)



dpo_model = AutoModelForCausalLM.from_pretrained('/home/bmh/project/llm_related/train_llm/model/save/dpo/dpo_model')
dpo_model.to(device)

pretrain_model.eval()
sft_model.eval()
dpo_model.eval()


def generate(mode, text_input, max_tokens=8192, temperature=0.85, top_p=0.85):
    messages = []
    if mode == "dpo":
        model = dpo_model
    elif mode == "sft":
        model = sft_model
    messages.append({"role": "user", "content": text_input})
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )[-max_tokens:]
    print(new_prompt)
    x = tokenizer(new_prompt).data['input_ids']
    x = (torch.tensor(x, dtype=torch.long, device=device)[None, ...])
    with torch.no_grad():
        res_y = model.generate(
            x,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False,
            rp=1.2,
            pad_token_id=tokenizer.pad_token_id
        )
        answer = tokenizer.decode(res_y.squeeze()[x.shape[1]:].tolist(), skip_special_tokens=True)
    return answer


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            mode = gr.Radio(["dpo", "sft"], label="选择模型")
            text_input = gr.Textbox(label="输入文本")
            text_output = gr.Textbox(label="输出文本")
            generate_button = gr.Button("生成")
            generate_button.click(generate, inputs=[mode,text_input], outputs=text_output)

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7891)

