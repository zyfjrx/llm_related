import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


# MoE
# 门控机制
class Gating(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.topk = config.topk
        self.expert_num = config.expert_num
        self.gate = nn.Linear(self.hidden_size, self.expert_num)

    def forward(self, x: torch.Tensor):
        logits = self.gate(x)
        logits_topk, indices = logits.topk(self.topk, dim=-1)
        zeros = torch.full_like(logits, float('-inf'))
        sparse_logits = zeros.scatter(dim=-1, index=indices, value=logits_topk)
        sparse_logits = F.softmax(sparse_logits, dim=-1)
        gate_logit = logits.view(-1, self.expert_num)
        return sparse_logits, indices, gate_logit
 #单个专家
class Expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)

    def forward(self, x: torch.Tensor):
        down_proj = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.expert_num)])
        self.gating = Gating(config)

    def forward(self, x: torch.Tensor):
        sparse_logits, indices, gate_logit = self.gating(x)
        final_outputs = torch.zeros_like(x)
        # 展平
        x_flat = x.view(-1, x.shape[-1])
        sparse_logits_flat = sparse_logits.view(-1, sparse_logits.shape[-1])

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(-1)
            expert_mask_flat = expert_mask.view(-1)
            if expert_mask_flat.any():
                expert_input = x_flat[expert_mask_flat]
                expert_output = expert(expert_input)

                gate_scores = sparse_logits_flat[expert_mask_flat,i].unsqueeze(1)
                weighted_output = expert_output * gate_scores
                final_outputs[expert_mask] += weighted_output

        return final_outputs,gate_logit
