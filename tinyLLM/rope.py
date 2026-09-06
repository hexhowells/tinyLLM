import torch
import torch.nn as nn


class RoPECache(nn.Module):
    def __init__(
            self, 
            head_dim: int, 
            max_seq_len: int, 
            base: float = 10000.0
        ):
        super().__init__()

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))

        pos_indicies = torch.arange(max_seq_len, dtype=torch.float32)

        freqs = torch.outer(pos_indicies, inv_freq)

        emb = torch.cat((freqs, freqs), dim=1)

        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)


    def forward(
            self, 
            seq_len: int, 
            dtype: torch.dtype = torch.float32
        ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype)
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]

    return torch.cat((-x2, x1), dim=1)


def apply_rotary_pos_emb(
        q: torch.Tensor, 
        k: torch.Tensor, 
        cos: torch.Tensor, 
        sin: torch.Tensor
    ):
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)

    return q_embed, k_embed