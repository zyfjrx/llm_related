import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple



# RoPE 旋转位置编码
def precompute_pos_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return pos_cis


def apply_rotary_emb(xq, xk, pos_cis):
    def unite_shape(pos_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert pos_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    pos_cis = unite_shape(pos_cis, xq_)
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int):
    bsz, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bsz, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(bsz, seq_len, n_kv_heads * n_rep, head_dim)
    )


class MQA(nn.Module):
    def __init__(self,
                 hidden_dim: int = 512,
                 n_heads: int = 8,
                 n_kv_heads: int = 1,
                 vocab_size: int = 6400,
                 max_seq_len: int = 512,
                 dropout: float = 0.0,
                 flash_attn: bool = False
                 ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)
        self.flash_attn = flash_attn
        self.head_dim = hidden_dim // n_heads
        self.max_seq_len = max_seq_len
        self.wq = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.wk = nn.Linear(self.hidden_dim, self.n_kv_heads*self.head_dim, bias=False)
        self.wv = nn.Linear(self.hidden_dim, self.n_kv_heads*self.head_dim, bias=False)
        self.wo = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        mask = torch.full((1, 1, self.max_seq_len, self.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self,
                x: torch.Tensor,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        # 应用旋转位置编码
        pos_cis = precompute_pos_cis(dim=self.head_dim, end=seq_len)
        xq, xk = apply_rotary_emb(xq, xk, pos_cis)

        # kv cache
        if past_key_value is not None:
            xk = torch.cat((past_key_value[0], xk), dim=1)
            xv = torch.cat((past_key_value[1], xv), dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )
        if self.flash_attn and seq_len != 1:
            dropout_p = 0.0
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=self.mask,
                dropout_p=dropout_p,
                is_causal=True
            )
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores += self.mask[:, :, :seq_len, :seq_len]
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.dropout(scores)
            output = scores @ xv
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.dropout(self.wo(output))
        return output, past_kv


if __name__ == '__main__':
    x = torch.randn(4, 100, 512)
    mha = MQA()
    output, past_kv = mha(x)
    print(output.shape)
    print(output)
