from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from mode_base import MiniMindLM
from config import LMConfig

# tokenizer = AutoTokenizer.from_pretrained('/home/bmh/project/llm_related/train_llm/model/save/pretrain/base')
# AutoConfig.register("minimind", LMConfig)
# AutoModelForCausalLM.register(LMConfig, MiniMindLM)
#
# model = AutoModelForCausalLM.from_pretrained('//home/bmh/project/llm_related/train_llm/model/save/pretrain/base')
#
# input_data = [tokenizer.bos_token_id] + tokenizer.encode('1+1等于')
# input_ids = torch.tensor(input_data).unsqueeze(0)
#
# for token in model.generate(input_ids):
#     print(tokenizer.decode(token))

x = torch.randn([4,2,5])
y = torch.randn([4,2,1])
print(x)
print(y)
print(x * y)
