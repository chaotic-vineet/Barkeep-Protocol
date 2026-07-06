import dataclasses
import math
import torch

from src.model_files.definitions import PositionalConfig, HyperParameters
from src.model_files.definitions import Linear

class Learned:
    """
    Generates a learnable positional encoding matrix of shape (vocab_dim, embedding_dim)
    Adds the positional information(P) to the embedded inputs(E[X]) as follows:
        encoded inputs = E[X] + P
        shape change: None, because it's just simple addition broadcasted over the batch_size
             (batch_size, context_size, embedding_dim) + (context_size, embedding_dim) =(batch_size, context_size, embedding_dim) + (1, context_size, embedding_dim)
    """
    def __init__(self, cfg: PositionalConfig, hp:HyperParameters):
        seed = hp.seed + (-101)
        self.positional_encoding_matrix = Linear(input_dim=hp.context_size, output_dim=hp.embedding_dim, seed=seed, bias=False, device=hp.device, scale=cfg.scale).weight
        self.cfg = cfg

    def __call__(self):
        return self.positional_encoding_matrix

    def parameters(self):
        return [self.positional_encoding_matrix]

    def config_dict(self):
        return dataclasses.asdict(self.cfg)

class Sinusoidal:
    def __init__(self, cfg: PositionalConfig, hp:HyperParameters):
        num_pairs = hp.embedding_dim // 2
        pairs = torch.arange(num_pairs, device=hp.device)
        pos = torch.arange(hp.context_size, device=hp.device)

        freqs = torch.exp( -2*pairs/hp.embedding_dim * math.log(10000))
        angles = torch.outer(pos, freqs)

        cos_values = torch.cos(angles)
        sin_values = torch.sin(angles)

        encoding = torch.stack((sin_values, cos_values), dim=-1).reshape(hp.context_size, hp.embedding_dim)

        self.positional_encoding_matrix = encoding
        self.cfg = cfg

    def __call__(self):
        return self.positional_encoding_matrix

    @staticmethod
    def parameters():
        return []

    def config_dict(self):
        return dataclasses.asdict(self.cfg)

class RoPE:
    def __init__(self, cfg: PositionalConfig, hp: HyperParameters, dim_k: int):
        num_pairs = dim_k // 2
        pairs = torch.arange(num_pairs, device=hp.device)
        pos = torch.arange(hp.context_size, device=hp.device)

        freqs = torch.exp((-2/dim_k)*pairs*math.log(cfg.theta))
        angles = torch.outer(pos, freqs)

        self.cos_table = torch.cos(angles)
        self.sin_table = torch.sin(angles)

        self.cfg = cfg

    def __call__(self, x:torch.tensor) -> torch.tensor:
        shape_in = x.shape
        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]

        out_even = x_even * self.cos_table - x_odd * self.sin_table
        out_odd  = x_even * self.sin_table + x_odd * self.cos_table

        out = torch.stack([out_even, out_odd], dim=-1).reshape(shape_in)

        return out
    
    @staticmethod
    def parameters():
        return []
    
    def config_dict(self):
        return dataclasses.asdict(self.cfg)

class ALiBi:
    def __init__(self, cfg: PositionalConfig, hp: HyperParameters, num_heads:int):
        positions = torch.arange(hp.context_size, device=hp.device)
        distance = positions.unsqueeze(1) - positions.unsqueeze(0)

        heads = torch.arange(num_heads, device=hp.device) + 1
        head_slopes = torch.exp((-8/num_heads)*heads*math.log(2))

        self.bias = -head_slopes[:, None, None] * distance[None, :, :]
        
        self.cfg = cfg

    def __call__(self, x:torch.tensor) -> torch.tensor:
        return x + self.bias
    
    @staticmethod
    def parameters():
        return []

    def config_dict(self):
        return dataclasses.asdict(self.cfg)

POSITIONAL_REGISTRY = {
    "rope": RoPE,
    "sinusoidal": Sinusoidal,
    "alibi": ALiBi,
    "learned": Learned,
}

def build_positional(cfg:PositionalConfig, hp:HyperParameters, num_heads:int|None, dim_k:int|None):
    cls = POSITIONAL_REGISTRY[cfg.kind]

    if cfg.kind in ["learned", "sinusoidal"]:
        return cls(cfg, hp)
    
    if cfg.kind == "rope":
        return cls(cfg, hp, dim_k)

    if cfg.kind == "alibi":
        return cls(cfg, hp, num_heads)
    
    else:
        raise ValueError("Unknown kind of positional method given.")