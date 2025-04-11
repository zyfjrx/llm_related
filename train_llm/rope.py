import torch


# 旋转矩阵
def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)

    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


if __name__ == '__main__':
    # freqs_cis = precompute_freqs_cis(8, 10)
    # print(freqs_cis)
    # print(freqs_cis.shape)

    tt = torch.tensor([1,3]).float()
    print(tt.shape)
    print(tt)
    ss = torch.view_as_complex(tt)
    print(ss)

