import torch
import math

from dataclasses import dataclass, field
from typing import Literal, Optional

@dataclass
class HyperParameters:
    device: str = 'xpu'
    seed: int = 42
    vocab_size: int = 3000
    context_size: int = 256
    embedding_dim: int = 128
    dropout_p: float = 0.2

@dataclass
class PositionalConfig:
    kind: str = "learned"
    theta: Optional[float] = None
    scale: Optional[float] = None

    def __post_init__(self):
        if self.kind == "rope":
            assert self.theta is not None, "RoPE requires theta to be given."
        if self.kind != "learned":
            assert self.scale is None, "Scale not needed for given positional encoding method."
        if self.kind == "learned":
            assert self.scale is not None, "Scale required for Learned positional encoding."


@dataclass
class FeedForwardConfig:
    kind: str = "non_gated"
    activation: str = "gelu"
    hidden_dim: int = 512
    gelu_implementation: Optional[str] = "approximate"
    alpha: Optional[float] = None

    def __post_init__(self):
        if self.activation == "leaky_relu":
            assert self.alpha is not None, "Leaky ReLU requires an alpha to be given."


score_functions = Literal["softmax", "relu_scores", "relu_kernel", "performer"]
mask_type = Literal["dense", "sliding_window", "sparse_block"]

@dataclass(frozen=True)
class AttentionConfig:
    num_heads: int = 4
    num_groups: int = 2
    dim_k: int = 16
    dim_v: int = 32
    mask: mask_type = "dense"
    window: Optional[int] = None
    block_size: Optional[int] = None
    num_local_blocks: Optional[int] = 1
    global_tokens: Optional[int] = None
    score_function: score_functions = "softmax"

    # TODO: Implement the __post_init__ checks to make sure that the attention variant is valid.
    def __post_init__(self):
        assert self.num_heads % self.num_groups == 0, "Number of heads must be divisible by number of groups."
        if self.mask == "sliding_window":
            assert self.window > 0, "A 0 window mask can't be given."
        if self.mask != "sliding_window":
            assert self.window is None, "Local attention needs a sliding window mask."
        if self.mask == "sparse_block":
            assert self.block_size is not None, "Sparse block needs a block size to be passed."

        if self.score_function == "relu_kernel" or self.score_function == "performer":
            assert self.mask == "dense"


    @property
    def kind(self) -> str:
        if self.num_groups == self.num_heads: return "mha"
        if self.num_groups == 1: return "mqa"
        return "gqa"

@dataclass
class NormalizationConfig:
    kind: str = "layer"
    bias: bool = False
    eps: float = 1e-5

@dataclass
class BlockConfig:
    num_blocks: int = 4
    norm_placement: str = "pre"
    residual: str = "sequential"

@dataclass
class ModelConfig: 
    positional: PositionalConfig = field(
        default_factory=PositionalConfig
    )
    feedforward: FeedForwardConfig = field(
        default_factory=FeedForwardConfig
    )
    attention: AttentionConfig = field(
        default_factory=AttentionConfig
    )
    normalization: NormalizationConfig = field(
        default_factory=NormalizationConfig
    )
    block: BlockConfig = field(
        default_factory=BlockConfig
    )
    hyperparameters: HyperParameters = field(
        default_factory=HyperParameters
    )
    def __post_init__(self):
        if self.attention.mask == "sparse_block":
            assert self.hyperparameters.context_size % self.attention.block_size == 0, "Context size must be divisible by block size for Sparse Block Attention."
        if self.positional.kind == "rope":
            assert self.attention.dim_k % 2 == 0, "RoPE needs an even query/key dimension."
        if self.positional.kind == "sinusoidal":
            assert self.hyperparameters.embedding_dim % 2 == 0, "Sinusoidal needs an even model dimension, i.e. embedding dimension."

@dataclass
class OptimizerConfig:
    kind: str = "adamw"
    beta_1: float = 0.9
    beta_2: float = 0.99
    eps: float = 1e-8
    decay: float = 0.1
    max_norm: float = 1.0

@dataclass
class TrainingConfig:
    batch_size: int = 128
    starting_lr: float = 3e-3
    ending_lr: float = 3e-4
    warmup_steps: int = 500
    num_steps: int = 5000

class Linear:
    """
    Returns a linear layer:
    y = x @ W + b

    by default, it adds bias: b
    """
    def __init__(self, input_dim, output_dim, seed, bias=True, device='xpu', scale=1.0):
        generator=torch.Generator(device=device).manual_seed(seed)

        self.weight = (
            torch.randn(size=(input_dim, output_dim), device=device, generator=generator)
            * 1/math.sqrt(input_dim)
            * scale
        ).requires_grad_(True)

        self.bias = torch.zeros(output_dim, device=device).requires_grad_(True) if bias else None

    def __call__(self, x):
        output = x @ self.weight
        if self.bias is not None:
             output = output + self.bias

        return output

    def parameters(self):
        return [self.weight] + ([self.bias] if self.bias is not None else [])
    
class Embedding:
    def __init__(self, hp:HyperParameters):
        seed = hp.seed + 1001
        self.embedding_matrix = Linear(input_dim=hp.vocab_size, output_dim=hp.embedding_dim, seed=seed, bias=False, device=hp.device, scale=0.1).weight

    def __call__(self, x):
        return self.embedding_matrix[x]
    
    def parameters(self):
        return [self.embedding_matrix]
    
    def config_dict(self):
        return {}

class Dropout:
    """
    Dropout class
    Drops the values in a given parameter with a probability:= p and scales the remaining values with 1/(1-p)
    """
    def __init__(self, p, flag: list|None=None):
        self.p = p
        self.flag = flag

    def __call__(self, x):
        if self.flag is None or self.flag[0] != 'Train':
            return x
        if self.p == 0:
            return x
        if self.p == 1:
            return torch.zeros_like(x)

        mask = torch.rand_like(x) > self.p
        return (x * mask) / (1.0 - self.p)

    @staticmethod
    def parameters():
        return []
    
    def config_dict(self):
        return {"dropout_p": self.p}