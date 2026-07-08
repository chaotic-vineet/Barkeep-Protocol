"""
A fully modular transformer architecture
"""
import dataclasses

import torch

from src.model_files.attention import build_attention
from src.model_files.definitions import Dropout, FeedForwardConfig, HyperParameters, ModelConfig
from src.model_files.definitions import Linear, Embedding
from src.model_files.activation import activation_function
from src.model_files.norm import build_norm
from src.model_files.positional import ALiBi, Learned, RoPE, Sinusoidal, build_positional

class FeedForwardNetwork:
    def __init__(self, cfg: FeedForwardConfig, hp: HyperParameters, layer_idx: int, scale: float):
        seed = hp.seed + 1001*layer_idx

        match cfg.kind:
            case "non_gated":
                self.weight_up   = Linear(input_dim=hp.embedding_dim, output_dim=cfg.hidden_dim, seed=seed+0, device=hp.device, scale=1, bias=True)
                self.weight_down = Linear(input_dim=cfg.hidden_dim, output_dim=hp.embedding_dim, seed=seed+1, device=hp.device, scale=scale, bias=True)
            
            case "gated":
                hidden_dim = int(2/3 * cfg.hidden_dim)
                hidden_dim = ((hidden_dim + 63) // 64) * 64 
                self.weight_gate = Linear(input_dim=hp.embedding_dim, output_dim=hidden_dim, seed=seed+0, device=hp.device, scale=1, bias=True)
                self.weight_up   = Linear(input_dim=hp.embedding_dim, output_dim=hidden_dim, seed=seed+1, device=hp.device, scale=1, bias=True)
                self.weight_down = Linear(input_dim=hidden_dim, output_dim=hp.embedding_dim, seed=seed+2, device=hp.device, scale=scale, bias=True)
        
        self.activation  = activation_function(cfg)
        self.cfg = cfg
        
    def __call__(self, x:torch.tensor) -> torch.tensor:
        match self.cfg.kind:
            case "non_gated":
                return self.weight_down(self.activation(self.weight_up(x)))
            
            case "gated":
                branch_1 = self.activation(self.weight_gate(x))
                branch_2 = self.weight_up(x)

                product = branch_1 * branch_2

                return self.weight_down(product)

    def parameters(self):
        match self.cfg.kind:
            case "non_gated":
                return [parameter for weight in [self.weight_up, self.weight_down] for parameter in weight.parameters()]

            case "gated":
                return [parameter for weight in [self.weight_up, self.weight_gate, self.weight_down] for parameter in weight.parameters()]

    def config_dict(self):
        return dataclasses.asdict(self.cfg)

# TODO: configurable sequential normalization paradigms. rather than 2 instances of same kind, we do 1 of each
class Block:
    def __init__(self, model_cfg: ModelConfig, hp:HyperParameters, layer_idx:int, flag:list, positional:RoPE|ALiBi|None, scale:float):
        self.attention = build_attention(model_cfg.attention, hp, layer_idx, flag, positional, scale)
        self.feedforward = FeedForwardNetwork(model_cfg.feedforward, hp, layer_idx, scale)
        self.norm1 = build_norm(model_cfg.normalization, hp)
        self.norm2 = build_norm(model_cfg.normalization, hp) if model_cfg.block.residual == "sequential" else None
        self.dropout = Dropout(hp.dropout_p, flag)
        self.cfg = model_cfg.block

    def __call__(self, x:torch.tensor) -> torch.tensor:
        match self.cfg.norm_placement:
            case "pre":
                match self.cfg.residual:
                    case "sequential":
                        x = x + self.dropout(self.attention(self.norm1(x)))
                        x = x + self.dropout(self.feedforward(self.norm2(x)))

                    case "parallel":
                        x = x + self.dropout(self.attention(self.norm1(x)) + self.feedforward(self.norm1(x)))
            
            case "post":
                match self.cfg.residual:
                    case "sequential":
                        x = self.norm1(x + self.dropout(self.attention(x)))
                        x = self.norm2(x + self.dropout(self.feedforward(x)))
                    
                    case "parallel":
                        x = self.norm1(x + self.dropout(self.attention(x) + self.feedforward(x)))

        return x

    def parameters(self):
        return [
            parameter
            for component in (
                [self.attention, self.feedforward, self.norm1, self.dropout] + ([self.norm2] if self.norm2 is not None else [])
            ) for parameter in component.parameters()
        ]

    def config_dict(self):
        return {
            "attention":      self.attention.config_dict(),
            "feedforward":    self.feedforward.config_dict(),
            "normalization":  self.norm1.config_dict(),
            "dropout":        self.dropout.config_dict(),
            "norm_placement": self.cfg.norm_placement,
            "residual":       self.cfg.residual,
        }
    

class Transformer:
    def __init__(self, model_cfg: ModelConfig):
        self.mode = ["Train"]

        block_scale = (2 * model_cfg.block.num_blocks) ** -0.5

        self.embedding = Embedding(model_cfg.hyperparameters)
        self.embedding_dropout = Dropout(model_cfg.hyperparameters.dropout_p, self.mode)
        
        self.positional = build_positional(model_cfg.positional, model_cfg.hyperparameters, model_cfg.attention.num_heads, model_cfg.attention.dim_k)
        
        self.blocks = [
            Block(model_cfg, model_cfg.hyperparameters, idx, self.mode, self.positional, block_scale)
            for idx in range(model_cfg.block.num_blocks)
        ]

        self.final_norm = build_norm(model_cfg.normalization, model_cfg.hyperparameters)

        self.cfg = model_cfg
        self.hp = model_cfg.hyperparameters
    
    def __call__(self, x:torch.tensor) -> torch.tensor:
        embedded = self.embedding(x)
        if isinstance(self.positional, (Learned, Sinusoidal)):
            encoded = embedded + self.positional()
        else:
            encoded = embedded
        
        representation = self.embedding_dropout(encoded)

        for block in self.blocks:
            representation = block(representation)

        normalized = self.final_norm(representation)
        logits = normalized @ self.embedding.embedding_matrix.T
        
        return logits
    
    def train(self):
        self.mode[0] = "Train" 

    def eval(self):
        self.mode[0] = "Eval"

    def parameters(self):
        block_parameters = []
        for block in self.blocks:
            block_parameters += block.parameters()
        return (
            self.embedding.parameters()
            + self.positional.parameters()
            + block_parameters
            + self.final_norm.parameters()
        )
    
    def config_dict(self):
        return {
            "positional": self.positional.config_dict(),
            "block":      self.blocks[0].config_dict(),
            "hp":         dataclasses.asdict(self.hp),
        }