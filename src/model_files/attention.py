import math
import dataclasses
import torch
from typing import Optional

from src.model_files.definitions import AttentionConfig, HyperParameters, ModelConfig
from src.model_files.definitions import Linear, Dropout
from src.model_files.positional import RoPE, ALiBi

def dense(context_size:int, device: str, **kwargs) -> torch.tensor:
    return torch.triu(
                torch.ones(
                    size=(context_size, context_size),
                    device=device,
                    dtype=torch.bool
                ),
                diagonal=1
            )

def sliding_window(context_size:int, device:str, window:int, **kwargs) -> torch.tensor:
    m = torch.ones(
        size=(context_size, context_size),
        device=device,
        dtype=torch.bool
    )

    return torch.triu(m, diagonal=1) | torch.tril(m, diagonal=-window)

def sparse_block(context_size:int, block_size:int, num_local_blocks:int, global_tokens:int, device:str, **kwargs) -> torch.tensor:
    num_blocks = context_size // block_size

    blocks = torch.ones(size=(num_blocks, num_blocks), dtype=torch.bool, device=device)
    allow_block = torch.tril(blocks, diagonal=0) & torch.triu(blocks, diagonal=-num_local_blocks)

    ones_blocks = torch.ones(size=(block_size, block_size), dtype=torch.bool, device=device)
    allowed = torch.kron(allow_block, ones_blocks)

    if global_tokens is not None:
        for global_token in range(global_tokens):
            allowed[global_token, :] = True
            allowed[:, global_token] = True

    causal = torch.tril(
        torch.ones(
            size=(context_size, context_size),
            device=device,
            dtype=torch.bool
        )
    )

    mask = ~(allowed & causal)
    
    return mask

MASKS = {
    "dense": dense,
    "sliding_window": sliding_window,
    "sparse_block": sparse_block
}


def softmax(masked_scores: torch.tensor) -> torch.tensor:
    max_vals = torch.max(masked_scores, dim=-1, keepdim=True).values

    scores = masked_scores - max_vals
    exp_scores = torch.exp(scores)
    sum_exp_scores = torch.sum(exp_scores, dim=-1, keepdim=True)

    return exp_scores / sum_exp_scores
    # return torch.nn.functional.softmax(masked_scores, dim=-1)

def relu_scores(masked_scores: torch.tensor) -> torch.tensor:
    masked_relu_scores = torch.where(masked_scores>0, masked_scores, 0)
    
    sum_relu_scores = torch.sum(masked_relu_scores, dim=-1, keepdim=True)

    clamped_scores = torch.clamp(sum_relu_scores, min=1e-9)
    
    return masked_relu_scores / clamped_scores

def phi_relu_kernel(x: torch.tensor, **kwargs) -> torch.tensor:
    return torch.where(x>0, x, 0)    

def phi_performer(x: torch.tensor, omega: torch.tensor, role:str) -> torch.tensor:
    m, _ = omega.shape
    projected = x @ omega.transpose(-2, -1)
    
    if role == "query":
        projected_max_vals = torch.max(projected, dim=-1, keepdim=True).values
    elif role == "key":
        projected_max_vals = torch.max(projected)

    normalizer = torch.exp(-0.5 * (x**2).sum(dim=-1, keepdim=True)) / (m**0.5)
    projected = projected - projected_max_vals

    return torch.exp(projected) * normalizer

SCORES = {
    "softmax":     softmax,
    "relu_scores": relu_scores,
}

def linear_attention_chunked(phi_Q, phi_K, V, chunk_size=64, eps=1e-9):
    B, h, T, m = phi_Q.shape
    d_v = V.shape[-1]

    S = torch.zeros(B, h, m, d_v, device=V.device, dtype=V.dtype)  
    z = torch.zeros(B, h, m,      device=V.device, dtype=V.dtype)

    outputs = []
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        q_c, k_c, v_c = phi_Q[:, :, start:end], phi_K[:, :, start:end], V[:, :, start:end]

        
        outer_c = k_c.unsqueeze(-1) * v_c.unsqueeze(-2)          
        KV_intra = torch.cumsum(outer_c, dim=2)
        z_intra  = torch.cumsum(k_c, dim=2)

        num   = torch.einsum('bhti,bhtij->bhtj', q_c, KV_intra) + q_c @ S      
        denom = (q_c * z_intra).sum(-1, keepdim=True) + (q_c @ z.unsqueeze(-1))
        outputs.append(num / torch.clamp(denom, min=eps))

        
        S = S + outer_c.sum(dim=2)
        z = z + k_c.sum(dim=2)

    return torch.cat(outputs, dim=2)

class Attention:
    def __init__(self, cfg: AttentionConfig, hp: HyperParameters, layer_idx: int, flag: list | None, positional: RoPE | ALiBi | None, scale:float):
        seed = hp.seed + 101*layer_idx

        self.query_weight = Linear(input_dim=hp.embedding_dim, output_dim=cfg.dim_k*cfg.num_heads, seed=seed+0, bias=False, device=hp.device, scale=1)
        self.key_weight   = Linear(input_dim=hp.embedding_dim, output_dim=cfg.dim_k*cfg.num_groups, seed=seed+1, bias=False, device=hp.device, scale=1)
        self.value_weight = Linear(input_dim=hp.embedding_dim, output_dim=cfg.dim_v*cfg.num_groups, seed=seed+2, bias=False, device=hp.device, scale=1)
        self.out_weight   = Linear(input_dim=cfg.dim_v*cfg.num_heads, output_dim=hp.embedding_dim, seed=seed+3, bias=False, device=hp.device, scale=scale)

        self.mask = MASKS[cfg.mask](device=hp.device, context_size=hp.context_size, **dataclasses.asdict(cfg))

        if cfg.score_function in ["relu_scores", "softmax"]:
            self.phi = None

        if cfg.score_function == "performer":
            generator = torch.Generator(device=hp.device).manual_seed(seed + 4)
            self.omega = torch.randn(size=(4*cfg.dim_k, cfg.dim_k), device=hp.device, generator=generator)
            self.phi = lambda x, role: phi_performer(x, self.omega, role)

        if cfg.score_function == "relu_kernel":
            self.phi = lambda x, role: phi_relu_kernel(x)

        self.attention_dropout = Dropout(p=hp.dropout_p, flag=flag) if flag is not None else None

        self.positional = positional

        self.cfg = cfg

    def __call__(self, x: torch.tensor):
        batch_size, context_size, *_ = x.shape

        Query = self.query_weight(x).reshape(batch_size, context_size, self.cfg.num_heads, self.cfg.dim_k).transpose(1, 2) # shape: (B, h, T, d_k) 
        Key   = self.key_weight(x).reshape(batch_size, context_size, self.cfg.num_groups, self.cfg.dim_k).transpose(1, 2) # shape: (B, g, T, d_k) 
        Value = self.value_weight(x).reshape(batch_size, context_size, self.cfg.num_groups, self.cfg.dim_v).transpose(1, 2) # shape: (B, g, T, d_v) 

        if isinstance(self.positional, RoPE):
            Query = self.positional(Query)
            Key   = self.positional(Key)

        num_repeats = self.cfg.num_heads//self.cfg.num_groups
        Key = Key.repeat_interleave(repeats=num_repeats, dim=1) # shape: (B, h, T, d_k) 
        Value = Value.repeat_interleave(repeats=num_repeats, dim=1) # shape: (B, h, T, d_v) 

        if self.cfg.score_function in ["softmax", "relu_scores"]:
            scaled_scores = (Query @ Key.transpose(-2, -1)) / math.sqrt(self.cfg.dim_k)
            
            if isinstance(self.positional, ALiBi):
                scaled_scores = self.positional(scaled_scores)
            
            masked_scores = scaled_scores.masked_fill(self.mask, float('-inf'))
            materialized = SCORES[self.cfg.score_function](masked_scores)

            if self.attention_dropout is not None:
                materialized = self.attention_dropout(materialized)

            out = (materialized @ Value)
        
        elif self.cfg.score_function in ["relu_kernel", "performer"]:
            phi_Query = self.phi(Query/(self.cfg.dim_k**0.25), role="query")
            phi_Key   = self.phi(Key/(self.cfg.dim_k**0.25), role="key")

            # outer_KV = phi_Key.unsqueeze(-1) * Value.unsqueeze(-2)

            # KV_cumsum = torch.cumsum(outer_KV, dim=2)
            # K_cumsum  = torch.cumsum(phi_Key, dim=2)

            # num = torch.einsum('bhti,bhtij->bhtj', phi_Query, KV_cumsum)
            # denom = (phi_Query * K_cumsum).sum(dim=-1, keepdim=True)
            # denom = torch.clamp(denom, min=1e-9)

            # out = num / denom
            out = linear_attention_chunked(phi_Query, phi_Key, Value)
        
        else:
            raise ValueError("Unknown score function used")
        
        out = out.transpose(1, 2).reshape(batch_size, context_size, self.cfg.num_heads*self.cfg.dim_v)

        return self.out_weight(out)
    
    def parameters(self):
        return [
            parameter for weight in [self.query_weight, self.key_weight, self.value_weight, self.out_weight] for parameter in weight.parameters()
        ]
    
    def config_dict(self):
        return dataclasses.asdict(self.cfg)
    
def build_attention(cfg: AttentionConfig, model_hp: HyperParameters, layer_idx: int, flag:list|None, positional:RoPE|ALiBi|None, scale):
    return Attention(cfg, model_hp, layer_idx, flag, positional, scale)