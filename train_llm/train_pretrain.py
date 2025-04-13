import os
import platform
import argparse
import time
import math
import warnings
import pandas as pd
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext

from transformers import AutoTokenizer

from train_llm.LMConfig import LMConfig
from train_llm.model_with_moe import MiniMindLM

warnings.filterwarnings('ignore')


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))



def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('model/minimind_tokenizer')
    model = MiniMindLM(lm_config).to(args.device)
    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer



# torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":

    lm_config = LMConfig()
    model, tokenizer = init_model(lm_config)
