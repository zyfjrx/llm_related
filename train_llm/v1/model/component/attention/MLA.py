from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


class MLA(nn.Module):
    def __init__(self,
                 dim,
                 n_heads,
                 q_lora_rank,
                 kv_lora_rank,
                 qk_nope_head_dim,
                 qk_rope_head_dim,
                 v_head_dim,
                 max_seq_len,
                 max_batch_size,
                 mode):
        super().__init__()
        self.dim = dim  # 隐藏层维度
        self.n_heads = n_heads  # 总头数
        self.q_lora_rank = q_lora_rank  # q低秩压缩到的维度
        self.kv_lora_rank = kv_lora_rank  # kv低秩压缩到的维度
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim  # qk的总维度，不带旋转位置编码的维度加上带旋转位置编码的维度
        self.v_head_dim = v_head_dim  # value的维度，等于不带旋转位置编码的k维度
        self.mode = mode
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.wq_a = nn.Linear(self.dim, self.q_lora_rank)  # q的降维矩阵
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim)  # q的升维矩阵
        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)  # kv的降维矩阵
        self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))  # kv的升维矩阵
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim)

    def forward(self,
                x: torch.Tensor,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False):
        bs, seq_len, _ = x.shape

        q = self.wq_b(self.wq_a(x))  # [bs, seq_len, n_heads * qk_head_dim]
        q = q.view(bs, seq_len, self.n_heads, self.qk_head_dim)  # [bs, n_heads,seq_len, qk_head_dim]
        # q_nope shape:[bs, n_heads, seq_len, qk_nope_head_dim] q_pe shape:[bs,  n_heads,seq_len, qk_rope_head_dim]
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.wkv_a(x)  # [bs, seq_len, kv_lora_rank + qk_rope_head_dim]
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(bs, seq_len, -1, self.qk_rope_head_dim)
        kv = (
            self.wkv_b(compressed_kv)
            .view(bs, seq_len, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        )
        kv_nope, value_status = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        pos_cis = precompute_pos_cis(dim=self.qk_rope_head_dim,end=seq_len)
        q_pe, k_pe = apply_rotary_emb(q_pe, k_pe, pos_cis)
        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat([kv_nope, repeat_kv(k_pe,self.n_heads)], dim=-1)


        q, k, v = (
            q.transpose(1, 2),
            k.transpose(1, 2),
            value_status.transpose(1, 2)
        )

        sources = (q @ k.transpose(-2, -1)) / math.sqrt(self.v_head_dim)
        sources = F.softmax(sources.float(), dim=-1).type_as(q)
        output = sources @ v
        output = output.transpose(1, 2)
        output = self.wo(output.reshape(bs, seq_len, -1))
        return output


if __name__ == '__main__':
    x = torch.randn(4, 512, 5120)

    dim = 5120
    n_heads = 128
    q_lora_rank = 1536
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 192
    max_seq_len = 512
    max_batch_size = 16
    mode = 'naive'

    mla = MLA(dim=dim,
              n_heads=n_heads,
              q_lora_rank=q_lora_rank,
              kv_lora_rank=kv_lora_rank,
              qk_nope_head_dim=qk_nope_head_dim,
              qk_rope_head_dim=qk_rope_head_dim,
              v_head_dim=v_head_dim,
              max_seq_len=max_seq_len,
              max_batch_size=max_batch_size,
              mode=mode)

    out = mla(x)
    print(out.shape)
