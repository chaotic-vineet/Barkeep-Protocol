import torch
import math
import dataclasses

from src.model_files.definitions import FeedForwardConfig, ModelConfig, HyperParameters

def GeLU(x:torch.tensor, method: str) -> torch.tensor:
    match method:
        case "exact":
            return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))
        case "approximate":
            return 0.5*x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        case _:
            raise ValueError("Unknown implementation method given.")

def SiLU(x:torch.tensor) -> torch.tensor:
    return x * torch.sigmoid(x)

def LeakyReLU(x:torch.tensor, alpha:float) -> torch.tensor:
    return torch.where(x>0, x, alpha*x)

ACTIVATION_REGISTRY = {
    "gelu" : GeLU,
    "silu" : SiLU,
    "leaky_relu" : LeakyReLU,
}

def activation_function(ffn_cfg: FeedForwardConfig):
    function = ACTIVATION_REGISTRY[ffn_cfg.activation]

    match ffn_cfg.activation:
        case "gelu":
            return lambda x: function(x, ffn_cfg.gelu_implementation)
        
        case "leaky_relu":
            return lambda x: function(x, ffn_cfg.alpha)

    return function