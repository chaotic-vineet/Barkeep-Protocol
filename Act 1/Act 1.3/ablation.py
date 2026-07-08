"""
Ablation runner for the 1.3 modular transformer study.

Designed to be called once on a cloud GPU (Kaggle / Colab) and produce
one JSONL file containing every run, comparable with analysis.py.

The study:
    baseline        learned + MHA + non-gated GELU + LayerNorm   (1.2 anchor)
    modern_full     RoPE   + GQA + gated SiLU      + RMSNorm
    modern -RoPE    strip one axis at a time back toward baseline
    modern -GQA
    modern -SwiGLU
    modern -RMSNorm

All runs share one TrainConfig, one SEED, one dataset build (BPE is
trained once on the train split and reused — the tokenizer is not a
variable in this study). Dropout masks draw from the global RNG, so
torch.manual_seed(SEED) is reset before each run: reruns of this script
on the same device reproduce exactly. Runs from DIFFERENT devices are
not comparable (different RNG streams, different kernels) — which is
why the baseline is re-run here instead of reusing the XPU number.

Usage (Kaggle):
    python ablation.py --data-dir /kaggle/input/wikitext2raw
Usage (Colab, data on Drive):
    python ablation.py --data-dir /content/drive/MyDrive/Wikitext-2

Optional:
    --runs modern_full baseline      run a subset
    --wandb-project Bar-eden-transformers
    --iterations 5000 --log-interval 500 (defaults shown)
"""

import argparse
import dataclasses
import gc
import json
import time

import torch

from src.datasets.dataset_wiki import make_wiki_dataset
from src.model_files.definitions import (
    AttentionConfig,
    BlockConfig,
    FeedForwardConfig,
    HyperParameters,
    ModelConfig,
    NormalizationConfig,
    PositionalConfig,
)
from src.modular_transformer import Transformer
from src.model_files.optimizer import AdamW
from src.train import TrainConfig, train_model

SEED = 42


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


# ---------------------------------------------------------------------------
# Run definitions
#
# Each entry is a function of (device, vocab_size, context_size) -> ModelConfig
# so the shared hyperparameters are written in exactly one place below.
# ---------------------------------------------------------------------------

# The four axes under study. Everything else (mask=dense, softmax,
# pre-norm, sequential residual, sizes) is held fixed across all runs.
LEARNED   = lambda: PositionalConfig(kind="learned", scale=0.1)
ROPE      = lambda: PositionalConfig(kind="rope", theta=10000.0)
MHA       = lambda: AttentionConfig(num_heads=4, num_groups=4, dim_k=16, dim_v=32)
GQA2      = lambda: AttentionConfig(num_heads=4, num_groups=2, dim_k=16, dim_v=32)
FFN_GELU  = lambda: FeedForwardConfig(kind="non_gated", activation="gelu", hidden_dim=512)
FFN_SWIGLU = lambda: FeedForwardConfig(kind="gated", activation="silu", hidden_dim=512)
LAYERNORM = lambda: NormalizationConfig(kind="layer", bias=False)
RMSNORM   = lambda: NormalizationConfig(kind="rms", bias=False)


def make_cfg(positional, attention, feedforward, normalization, hp) -> ModelConfig:
    return ModelConfig(
        positional=positional(),
        attention=attention(),
        feedforward=feedforward(),
        normalization=normalization(),
        block=BlockConfig(num_blocks=4, norm_placement="pre", residual="sequential"),
        hyperparameters=hp,
    )


RUNS = {
    # name:                (positional, attention, feedforward, normalization)
    "baseline":            (LEARNED, MHA,  FFN_GELU,   LAYERNORM),
    "modern_full":         (ROPE,    GQA2, FFN_SWIGLU, RMSNORM),
    "modern_minus_rope":   (LEARNED, GQA2, FFN_SWIGLU, RMSNORM),
    "modern_minus_gqa":    (ROPE,    MHA,  FFN_SWIGLU, RMSNORM),
    "modern_minus_swiglu": (ROPE,    GQA2, FFN_GELU,   RMSNORM),
    "modern_minus_rms":    (ROPE,    GQA2, FFN_SWIGLU, LAYERNORM),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="dir containing wiki.{train,valid,test}.raw")
    parser.add_argument("--runs", nargs="*", default=list(RUNS), choices=list(RUNS))
    parser.add_argument("--device", default=None)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=3000)
    parser.add_argument("--context-size", type=int, default=256)
    parser.add_argument("--log-path", default="runs_ablation.jsonl")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    args = parser.parse_args()

    device = args.device or pick_device()
    print(f"device: {device}")

    train_config = TrainConfig(
        lr_start=3e-3,
        lr_end=3e-4,
        iterations=args.iterations,
        log_interval=args.log_interval,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        log_path=args.log_path,
        max_norm=1.0,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )

    print("building dataset (BPE trains once, shared by every run)...")
    t = time.time()
    data = make_wiki_dataset(
        data_dir=args.data_dir,
        device=device,
        context_size=args.context_size,
        vocab_size=args.vocab_size,
    )
    print(f"dataset ready in {time.time() - t:.1f}s | "
          f"train chunks: {data['train'].inputs.shape[0]:,}")

    summary = []

    for name in args.runs:
        positional, attention, feedforward, normalization = RUNS[name]
        hp = HyperParameters(
            device=device,
            seed=SEED,
            vocab_size=args.vocab_size,
            context_size=args.context_size,
            embedding_dim=128,
            dropout_p=0.2,
        )
        cfg = make_cfg(positional, attention, feedforward, normalization, hp)

        torch.manual_seed(SEED)  # dropout masks draw from the global RNG

        model = Transformer(cfg)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"\n=== {name} | {num_params:,} params ===")

        optimizer = AdamW(
            model.parameters(),
            betas=(0.9, 0.99),
            eps=1e-8,
            weight_decay=0.1,
        )

        results = train_model(
            SEED=SEED,
            model=model,
            optimizer=optimizer,
            train=data["train"],
            dev=data["dev"],
            config=train_config,
            run_name=name,
            device=device,
            pause_time=0.0,  # cloud GPUs don't need thermal pauses
        )

        best_dev = min(results["dev_losses"])
        best_step = results["dev_itrns"][results["dev_losses"].index(best_dev)]
        summary.append({
            "run_name": name,
            "num_params": num_params,
            "final_dev": results["dev_losses"][-1],
            "best_dev": best_dev,
            "best_step": best_step,
            "elapsed_s": results["elapsed"],
        })

        del model, optimizer, results
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 78)
    header = f"{'run':<22}{'params':>12}{'final dev':>12}{'best dev':>12}{'@step':>8}{'time':>9}"
    print(header)
    print("-" * 78)
    for row in summary:
        print(f"{row['run_name']:<22}{row['num_params']:>12,}"
              f"{row['final_dev']:>12.4f}{row['best_dev']:>12.4f}"
              f"{row['best_step']:>8,}{row['elapsed_s']:>8.0f}s")

    if train_config.log_path is not None:
        with open(train_config.log_path, "a") as f:
            f.write(json.dumps({"type": "summary", "rows": summary}) + "\n")
        print(f"\nfull curves in {train_config.log_path}")


if __name__ == "__main__":
    main()
