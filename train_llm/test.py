from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from model.model import MiniMindLM
from model.config import LMConfig

tokenizer = AutoTokenizer.from_pretrained('/home/bmh/project/llm_related/train_llm/model/save/full_sft/minimind_base_sft')
AutoConfig.register("minimind", LMConfig)
AutoModelForCausalLM.register(LMConfig, MiniMindLM)

model = AutoModelForCausalLM.from_pretrained('/home/bmh/project/llm_related/train_llm/model/save/full_sft/minimind_base_sft')

# input_data = [tokenizer.bos_token_id] + tokenizer.encode('1+1等于多少')
# input_ids = torch.tensor(input_data).unsqueeze(0)

input_data = tokenizer.apply_chat_template([{'role':'user', 'content':'1+1等于多少？'}])
input_ids = torch.tensor(input_data).unsqueeze(0)

outputs = model.generate(
                input_ids,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=1024,
                temperature=0.85,
                top_p=0.85,
                stream=False,
                pad_token_id=tokenizer.pad_token_id
            )
token = outputs.squeeze()[input_ids.shape[1]:].tolist()
print(tokenizer.decode(token,skip_special_tokens=True))
# for token in model.generate(input_ids):
#     print(tokenizer.decode(token))

# tokenizer = AutoTokenizer.from_pretrained('/home/bmh/project/llm_related/train_llm/model/save/pretrain/minimind_base_pre/')
# model = MiniMindLM(LMConfig())
# ckp = "/home/bmh/project/llm_related/train_llm/model/save/pretrain/minimind_base_pre/pytorch_model.bin"
# state_dict = torch.load(ckp, map_location="cuda")
# model.load_state_dict(state_dict)
# input_data = [tokenizer.bos_token_id] + tokenizer.encode('马克思主义基本原理')
# input_ids = torch.tensor(input_data).unsqueeze(0)

