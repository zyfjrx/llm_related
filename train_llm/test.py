from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from model.model import MiniMindLM
from model.config import LMConfig

# tokenizer = AutoTokenizer.from_pretrained('/home/bmh/project/llm_related/train_llm/model/save/pretrain/minimind_base_pre')
# AutoConfig.register("minimind", LMConfig)
# AutoModelForCausalLM.register(LMConfig, MiniMindLM)
#
# model = AutoModelForCausalLM.from_pretrained('/home/bmh/project/llm_related/train_llm/model/save/pretrain/minimind_base_pre')
#
# input_data = [tokenizer.bos_token_id] + tokenizer.encode('马克思主义基本原理')
# input_ids = torch.tensor(input_data).unsqueeze(0)
#
# for token in model.generate(input_ids):
#     print(tokenizer.decode(token))

tokenizer = AutoTokenizer.from_pretrained('/home/bmh/project/llm_related/train_llm/model/save/pretrain/minimind_base_pre/')
model = MiniMindLM(LMConfig())
ckp = "/home/bmh/project/llm_related/train_llm/model/save/pretrain/minimind_base_pre/pytorch_model.bin"
state_dict = torch.load(ckp, map_location="cuda")
model.load_state_dict(state_dict)
input_data = [tokenizer.bos_token_id] + tokenizer.encode('马克思主义基本原理')
input_ids = torch.tensor(input_data).unsqueeze(0)

