import dataclasses
import torch

from src.model_files.definitions import ModelConfig, HyperParameters, NormalizationConfig

class LayerNorm:
    """
    Layer normalization over the last dimension.
    (B, T, d) → (B, T, d).

    Population variance (divides by d, not d-1).
    gamma=1, beta=0 at init (identity transform).
    """
    def __init__(self, cfg: NormalizationConfig, hp: HyperParameters):
        self.gamma = torch.ones(hp.embedding_dim, device=hp.device).requires_grad_(True)
        self.beta = torch.zeros(hp.embedding_dim, device=hp.device).requires_grad_(True) if cfg.bias else None
        self.eps = cfg.eps
        self.cfg = cfg

    def __call__(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x_hat = (x - mean) / (var + self.eps) ** 0.5
        return self.gamma * x_hat + (self.beta if self.beta is not None else 0)

    def parameters(self):
        return [self.gamma] + ([self.beta] if self.beta is not None else [])

    def config_dict(self):
        return dataclasses.asdict(self.cfg)

class RMSNorm:
    """
    RMS normalization over the last dimension.
    (B, T, d) → (B, T, d).
    """
    def __init__(self, cfg: NormalizationConfig, hp: HyperParameters):
        self.gamma = torch.ones(hp.embedding_dim, device=hp.device).requires_grad_(True)
        self.beta = torch.zeros(hp.embedding_dim, device=hp.device).requires_grad_(True) if cfg.bias else None
        self.eps = cfg.eps
        self.cfg = cfg

    def __call__(self, x):
        rms = ((x.pow(2)).mean(-1, keepdim=True) + self.eps).sqrt()
        x_hat = x / rms

        return self.gamma * x_hat + (self.beta if self.beta is not None else 0)

    def parameters(self):
        return [self.gamma] + ([self.beta] if self.beta is not None else [])

    def config_dict(self):
        return dataclasses.asdict(self.cfg)

class ScaleNorm:
    """
    Scale normalization over the last dimension.
    (B, T, d) → (B, T, d).
    """
    def __init__(self, cfg: NormalizationConfig, hp: HyperParameters):
        self.gamma = torch.full(size=(), fill_value=hp.embedding_dim**0.5, device=hp.device).requires_grad_(True)
        self.beta = torch.zeros((), device=hp.device).requires_grad_(True) if cfg.bias else None
        self.eps = cfg.eps
        self.cfg = cfg

    def __call__(self, x):
        l2_normalizer = (x.pow(2).sum(-1, keepdim=True) + self.eps).sqrt()
        x_hat = x / l2_normalizer

        return self.gamma * x_hat + (self.beta if self.beta is not None else 0)

    def parameters(self):
        return [self.gamma] + ([self.beta] if self.beta is not None else [])

    def config_dict(self):
        return dataclasses.asdict(self.cfg)

NORM_REGISTRY = {
    "layer": LayerNorm,
    "rms": RMSNorm,
    "scale": ScaleNorm
}

def build_norm(cfg:NormalizationConfig, hp: HyperParameters) -> LayerNorm|RMSNorm|ScaleNorm:
    normalization_class = NORM_REGISTRY[cfg.kind]
    return normalization_class(cfg, hp)