"""
Ablation runner v2 for the 1.3 modular transformer study.

Changes from v1:
    - multiple seeds per configuration (--seeds), run named {config}_s{seed}
    - saves each trained model's weights   -> <out-root>/Model Instances/{run}.pt
    - saves each run's full results dict   -> <out-root>/Training Results/{run}.json
    - saves the full diagnostic plot suite -> <out-root>/Plots/{run}/
    - evaluates FINAL loss on the ENTIRE dev split (batched, no sampling),
      so the headline number carries zero dev-batch noise
    - default batch size 64 (v1 used 128; v2 numbers are NOT comparable
      to v1 numbers — different tokens seen, different noise scale)

Per-checkpoint dev losses inside train_model still use that run's
seeded dev batch, so curves across seeds carry some dev-sampling
noise; the full-split evaluation at the end is the comparison number.

The diagnostic probe batch (16 dev rows) is drawn ONCE with a fixed
seed and shared by every run, so cross-run activation/gradient
histograms are comparable — differences are the architecture, not
the inputs.

Usage (Kaggle, from repo root):
    PYTHONPATH=. python "Act 1/Act 1.3/ablation.py" --data-dir /kaggle/input/<ds>
"""

import argparse
import gc
import json
import os
import time

import torch

from src.datasets.dataset_wiki import make_wiki_dataset, WikiDataset
from src.model_files.definitions import (
    AttentionConfig,
    BlockConfig,
    FeedForwardConfig,
    HyperParameters,
    ModelConfig,
    NormalizationConfig,
    OptimizerConfig,
    PositionalConfig,
    TrainingConfig,
)
from src.modular_transformer import Transformer
from src.model_files.optimizer import AdamW
from src.train import train_model, build_run_config
from src.analysis import get_named_parameters, run_all_diagnostics


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


# ---------------------------------------------------------------------------
# Run definitions — the four axes under study. Everything else (dense mask,
# softmax, pre-norm, sequential residual, sizes) is held fixed.
# ---------------------------------------------------------------------------

LEARNED    = lambda: PositionalConfig(kind="learned", scale=0.1)
ROPE       = lambda: PositionalConfig(kind="rope", theta=10000.0)
MHA        = lambda: AttentionConfig(num_heads=4, num_groups=4, dim_k=16, dim_v=32)
GQA2       = lambda: AttentionConfig(num_heads=4, num_groups=2, dim_k=16, dim_v=32)
FFN_GELU   = lambda: FeedForwardConfig(kind="non_gated", activation="gelu", hidden_dim=512)
FFN_SWIGLU = lambda: FeedForwardConfig(kind="gated", activation="silu", hidden_dim=512)
LAYERNORM  = lambda: NormalizationConfig(kind="layer", bias=False)
RMSNORM    = lambda: NormalizationConfig(kind="rms", bias=False)

OPTIMIZER = OptimizerConfig(kind="adamw", betas=(0.9, 0.99), eps=1e-8, weight_decay=0.1, max_norm=1.0)

RUNS = {
    # name:                (positional, attention, feedforward, normalization)
    "baseline":            (LEARNED, MHA,  FFN_GELU,   LAYERNORM),
    "modern_full":         (ROPE,    GQA2, FFN_SWIGLU, RMSNORM),
    "modern_minus_rope":   (LEARNED, GQA2, FFN_SWIGLU, RMSNORM),
    "modern_minus_gqa":    (ROPE,    MHA,  FFN_SWIGLU, RMSNORM),
    "modern_minus_swiglu": (ROPE,    GQA2, FFN_GELU,   RMSNORM),
    "modern_minus_rms":    (ROPE,    GQA2, FFN_SWIGLU, LAYERNORM),
}


def make_cfg(positional, attention, feedforward, normalization, hp) -> ModelConfig:
    return ModelConfig(
        positional=positional(),
        attention=attention(),
        feedforward=feedforward(),
        normalization=normalization(),
        block=BlockConfig(num_blocks=4, norm_placement="pre", residual="sequential"),
        hyperparameters=hp,
    )


def save_model(model, path, run_config):
    """
    Named state dict (get_named_parameters order) + the run's full config
    record, so the model can be reconstructed and reloaded later:
        state = torch.load(path); model = Transformer(<config rebuilt from record>)
        for (name, p) in get_named_parameters(model): p.data.copy_(state["state"][name])
    """
    state = {name: p.detach().cpu() for name, p in get_named_parameters(model)}
    torch.save({"state": state, "run_config": run_config}, path)


def full_dev_loss(model, dev, batch_size, device):
    """
    Cross-entropy over the ENTIRE dev split, batched, no sampling —
    the zero-dev-noise comparison number. Weighted by rows so the
    ragged last batch doesn't skew the mean.
    """
    model.eval()
    total, rows = 0.0, 0
    with torch.no_grad():
        for start in range(0, dev.inputs.shape[0], batch_size):
            xb = dev.inputs[start:start + batch_size]
            yb = dev.targets[start:start + batch_size]
            logits = model(xb)
            loss = torch.nn.functional.cross_entropy(
                logits.permute(0, 2, 1), yb, ignore_index=dev.pad_idx)
            total += loss.item() * xb.shape[0]
            rows += xb.shape[0]
    model.train()
    return total / rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="dir containing wiki.{train,valid,test}.raw")
    parser.add_argument("--runs", nargs="*", default=list(RUNS), choices=list(RUNS))
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    parser.add_argument("--device", default=None)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=3000)
    parser.add_argument("--context-size", type=int, default=256)
    parser.add_argument("--out-root", default=os.path.join("Act 1", "Act 1.3"),
                        help="root for Model Instances/, Training Results/, Plots/")
    parser.add_argument("--log-path", default=None,
                        help="defaults to <out-root>/runs_ablation_v2.jsonl")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--no-diagnostics", action="store_true",
                        help="skip the per-run plot suite")
    args = parser.parse_args()

    device = args.device or pick_device()
    print(f"device: {device}")

    models_dir  = os.path.join(args.out_root, "Model Instances")
    results_dir = os.path.join(args.out_root, "Training Results")
    plots_dir   = os.path.join(args.out_root, "Plots")
    for d in (models_dir, results_dir, plots_dir):
        os.makedirs(d, exist_ok=True)

    log_path = args.log_path or os.path.join(args.out_root, "runs_ablation_v2.jsonl")

    train_config = TrainingConfig(
        lr_start=3e-3,
        lr_end=3e-4,
        iterations=args.iterations,
        log_interval=args.log_interval,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        log_path=log_path,
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
          f"train chunks: {data['train'].inputs.shape[0]:,} | "
          f"dev chunks: {data['dev'].inputs.shape[0]:,}")

    # one probe batch, fixed seed, shared by EVERY run — cross-run
    # diagnostic plots then differ only by architecture, not inputs
    probe_gen = torch.Generator(device=device).manual_seed(4242)
    probe_idx = torch.randint(0, data["dev"].inputs.shape[0], (16,),
                              device=device, generator=probe_gen)
    probe_x = data["dev"].inputs[probe_idx]
    probe_y = data["dev"].targets[probe_idx]

    summary = []
    n_total = len(args.runs) * len(args.seeds)
    done = 0

    for name in args.runs:
        positional, attention, feedforward, normalization = RUNS[name]

        for seed in args.seeds:
            run_name = f"{name}_s{seed}"
            done += 1

            hp = HyperParameters(
                device=device,
                seed=seed,
                vocab_size=args.vocab_size,
                context_size=args.context_size,
                embedding_dim=128,
                dropout_p=0.2,
            )
            cfg = make_cfg(positional, attention, feedforward, normalization, hp)

            torch.manual_seed(seed)  # dropout masks draw from the global RNG

            model = Transformer(cfg)
            num_params = sum(p.numel() for p in model.parameters())
            print(f"\n=== [{done}/{n_total}] {run_name} | {num_params:,} params ===")

            optimizer = AdamW(parameters=model.parameters(), cfg=OPTIMIZER)

            results = train_model(
                SEED=seed,
                model=model,
                optimizer=optimizer,
                train=data["train"],
                dev=data["dev"],
                config=train_config,
                run_name=run_name,
                device=device,
                pause_time=0.0,
            )

            # zero-dev-noise headline number: the whole dev split
            dev_full = full_dev_loss(model, data["dev"], args.batch_size, device)
            print(f"  full-dev loss: {dev_full:.4f} "
                  f"(sampled-batch final: {results['dev_losses'][-1]:.4f})")

            run_config = build_run_config(run_name, model, train_config)

            save_model(model, os.path.join(models_dir, f"{run_name}.pt"), run_config)

            results["dev_full"] = dev_full
            with open(os.path.join(results_dir, f"{run_name}.json"), "w") as f:
                json.dump(results, f)

            if not args.no_diagnostics:
                run_all_diagnostics(model, results, probe_x, probe_y,
                                    model_name=run_name, out_dir=plots_dir)

            summary.append({
                "run_name": run_name,
                "config": name,
                "seed": seed,
                "num_params": num_params,
                "dev_full": dev_full,
                "final_dev_sampled": results["dev_losses"][-1],
                "best_dev_sampled": min(results["dev_losses"]),
                "elapsed_s": results["elapsed"],
            })

            del model, optimizer, results
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    # per-config aggregation across seeds
    print("\n" + "=" * 84)
    header = (f"{'config':<22}{'params':>12}{'dev_full mean':>15}"
              f"{'min':>9}{'max':>9}{'spread':>9}{'seeds':>7}")
    print(header)
    print("-" * 84)
    for name in args.runs:
        rows = [r for r in summary if r["config"] == name]
        if not rows:
            continue
        vals = [r["dev_full"] for r in rows]
        mean = sum(vals) / len(vals)
        print(f"{name:<22}{rows[0]['num_params']:>12,}{mean:>15.4f}"
              f"{min(vals):>9.4f}{max(vals):>9.4f}{max(vals) - min(vals):>9.4f}"
              f"{len(vals):>7}")

    with open(log_path, "a") as f:
        f.write(json.dumps({"type": "summary", "rows": summary}) + "\n")
    print(f"\ncurves: {log_path}\nweights: {models_dir}\nresults: {results_dir}\nplots: {plots_dir}")


if __name__ == "__main__":
    main()