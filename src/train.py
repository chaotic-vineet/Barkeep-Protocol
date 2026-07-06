"""
Hand-rolled training loop for the Act 1.2 attention models — no
torch.optim. Parameters are updated in place by an optimizer object
(e.g. the hand-rolled AdamW) called with the current learning rate,
which follows linear warmup into a cosine schedule.

Works with any model exposing __call__, parameters(), and config_dict()
(see model.py), and any optimizer exposing __call__(lr) and
step_magnitudes(lr).

Logging (both optional, independent):
    JSONL — if config.log_path is set, train_model writes JSONL records:
        - one "config" record at the start of the run: model.config_dict(),
          total parameter count, and config.__dict__
        - one "metric" record every config.log_interval steps: step,
          train_loss, dev_loss, lr, grad_norm, clipped
    Weights & Biases — if config.wandb_project is set, the same config
        record initializes a wandb run (named run_name) and the same
        metric fields are logged at every checkpoint, plus the mean
        log10 update/data ratio across matrix parameters. wandb is
        imported lazily so the dependency is only required when used.
        Set WANDB_MODE=offline to log locally without an account.

Multiple runs can share one log file and be compared via
analysis.load_runs / analysis.plot_run_comparison; on wandb, runs
sharing a project are compared in the workspace UI.
"""

import torch
import time
import math
import json
from dataclasses import dataclass

@dataclass
class TrainConfig:
    """
    Shared training hyperparameters. For an ablation, reuse one
    TrainConfig unchanged across runs — only run_name and the model
    should differ.

    lr warms up linearly for `warmup_steps`, then follows a cosine
    schedule from lr_start down to lr_end over the remaining steps.
    Dev loss is evaluated every `log_interval` steps on a fixed dev
    batch drawn once per run. log_path=None disables JSONL logging;
    wandb_project=None disables W&B logging.

    max_norm is the global gradient-clipping threshold and is owned
    here (not in OptimizerConfig) because clipping is implemented in
    the training loop.
    """
    lr_start: float
    lr_end: float
    iterations: int
    log_interval: int
    warmup_steps: int
    batch_size: int = 4096
    log_path: str | None = None
    max_norm: float = 1.0
    wandb_project: str | None = None
    wandb_entity: str | None = None

def build_run_config(run_name, model, config):
    """
    One flat dict describing this run: model.config_dict(), total
    parameter count, and every TrainConfig field. Shared by the JSONL
    "config" record and wandb.init(config=...). Asserts the two
    namespaces don't collide, since both flatten into one record.
    """
    config_fields = config.__dict__
    model_fields = model.config_dict()
    shared_keys = set(config_fields) & set(model_fields)
    assert not shared_keys, f"config/model key collision in log record: {shared_keys}"

    return {
        "run_name": run_name,
        "num_params": sum(p.numel() for p in model.parameters()),
        **config_fields,
        **model_fields,
    }

def log_run_config(log_path, run_config):
    """
    Write the run_config dict as one JSONL "config" record. No-op if
    log_path is None.
    """
    if log_path is None:
        return

    record = {"type": "config", **run_config}
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def init_wandb(config, run_name, run_config):
    """
    Start a wandb run if config.wandb_project is set, else return None.
    Lazy import: wandb is only required when actually used.
    """
    if config.wandb_project is None:
        return None

    import wandb
    return wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=run_name,
        config=run_config,
    )


def train_model(SEED, model, optimizer, train, dev, config, run_name, device, pause_time):
    """
    Train `model` on `train` for config.iterations steps, evaluating on
    a fixed dev batch every config.log_interval steps.

    Batches are drawn from a generator seeded with `SEED`, created
    fresh inside this function — so every call sees an identical batch
    sequence regardless of how much random state model construction
    consumed beforehand. The dev batch is drawn once (from an equally
    seeded generator) and reused at every checkpoint, so dev-curve
    movement reflects the model, not dev-batch resampling.

    Thermal pauses (pause_time minutes per checkpoint) are excluded
    from all reported timings.

    If config.log_path is set, writes one "config" record (see
    log_run_config) and one "metric" record per checkpoint. If
    config.wandb_project is set, mirrors both to Weights & Biases.

    Returns a dict:
        lr_per_itrn:        list[float], length config.iterations
        loss_per_itrn:      list[float], length config.iterations
        grad_norm_per_itrn: list[float], pre-clip global grad norm per step
        dev_losses:         list[float], one per checkpoint
        dev_itrns:          list[int],   step numbers of each checkpoint
        checkpoint_times:   list[float], active seconds elapsed at each checkpoint
        ud:                 list[list[float]], log10 update/data ratio per
                            checkpoint, matrix parameters (ndim >= 2) only
        ud_param_indices:   list[int], indices into model.parameters() that
                            the ud columns correspond to
        elapsed:            float, active wall-clock seconds (sleeps excluded)
    """
    run_config = build_run_config(run_name, model, config)
    log_run_config(config.log_path, run_config)
    wandb_run = init_wandb(config, run_name, run_config)

    batch_generator = torch.Generator(device=device).manual_seed(SEED)
    dev_batch_generator = torch.Generator(device=device).manual_seed(SEED)

    lr_start = config.lr_start
    lr_end = config.lr_end

    lr_per_itrn        = []
    checkpoint_times   = []
    loss_per_itrn      = torch.zeros(config.iterations, device=device)
    grad_norm_per_itrn = torch.zeros(config.iterations, device=device)
    dev_losses         = []
    dev_itrns          = []
    ud                 = []
    total_sleep        = 0.0

    if 'xpu' in device:
        torch.xpu.synchronize()
    if 'cuda' in device:
        torch.cuda.synchronize()
    t0 = time.time()

    parameters = list(model.parameters())

    # update/data ratios are only meaningful for weight matrices:
    # 0-dim params (ScaleNorm gamma) have NaN std, and all-ones /
    # all-zeros vectors (norm gammas, biases) have zero std.
    matrix_indices = [i for i, p in enumerate(parameters) if p.ndim >= 2]

    max_norm = config.max_norm

    warmup_steps = config.warmup_steps
    iterations = config.iterations
    batch_size = config.batch_size

    train_inputs_shape = train.inputs.shape[0]
    dev_inputs_shape = dev.inputs.shape[0]
    pad_idx = train.pad_idx

    # fixed dev batch: drawn once, reused at every checkpoint
    dev_idx = torch.randint(0, dev_inputs_shape, (batch_size,), device=device, generator=dev_batch_generator)
    dev_inputs  = dev.inputs[dev_idx]
    dev_targets = dev.targets[dev_idx]

    for itrn in range(1, iterations+1):
        idx = torch.randint(0, train_inputs_shape, (batch_size,), device=device, generator=batch_generator)

        logits = model(train.inputs[idx])
        loss = torch.nn.functional.cross_entropy(logits.permute(0, 2, 1), train.targets[idx], ignore_index=pad_idx)

        if itrn < warmup_steps:
            lr = lr_start * itrn / warmup_steps
        else:
            cosine_itrn = itrn - warmup_steps
            cosine_itrns = max(iterations - warmup_steps, 1)
            lr = lr_end + 0.5 * (lr_start - lr_end) * (1 + math.cos(math.pi * cosine_itrn / cosine_itrns))

        for parameter in parameters:
            parameter.grad = None
        loss.backward()

        global_norm_squared = 0
        for parameter in parameters:
            global_norm_squared += parameter.grad.pow(2).sum()

        global_norm = global_norm_squared ** 0.5
        clip_coeff = torch.clamp(max_norm / (global_norm + 1e-8), max=1.0)

        for parameter in parameters:
            parameter.grad = parameter.grad * clip_coeff

        optimizer(lr)

        lr_per_itrn.append(lr)
        loss_per_itrn[itrn-1] = loss.detach()
        grad_norm_per_itrn[itrn-1] = global_norm.detach()

        if itrn % config.log_interval == 0:
            sleep_start = time.time()
            time.sleep(pause_time*60)
            total_sleep += time.time() - sleep_start

            model.eval()

            with torch.no_grad():
                dev_logits = model(dev_inputs)
                dev_loss = torch.nn.functional.cross_entropy(dev_logits.permute(0, 2, 1), dev_targets, ignore_index=pad_idx)

            trn_loss_val = loss.item()
            dev_loss_val = dev_loss.item()
            grad_norm_val = global_norm.item()

            dev_losses.append(dev_loss_val)
            dev_itrns.append(itrn)

            print(f"  step {itrn:>7,} | dev {dev_loss_val:.4f} vs train {trn_loss_val:.4f} | lr {lr:.4f} | grad norm {grad_norm_val:.3f}")

            magnitudes = optimizer.step_magnitudes(lr)

            ud.append([
                (magnitudes[i].std() / parameters[i].data.std()).log10().item()
                for i in matrix_indices
            ])

            if 'xpu' in device:
                torch.xpu.synchronize()
            if 'cuda' in device:
                torch.cuda.synchronize()
            checkpoint_times.append(time.time() - t0 - total_sleep)

            metric = {
                "step": itrn,
                "train_loss": trn_loss_val,
                "dev_loss": dev_loss_val,
                "lr": lr,
                "grad_norm": grad_norm_val,
                "clipped": grad_norm_val > max_norm,
            }

            if config.log_path is not None:
                record = {"type": "metric", "run_name": run_name, **metric}

                with open(config.log_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

            if wandb_run is not None:
                wandb_run.log(
                    {**metric, "ud_mean": sum(ud[-1]) / len(ud[-1])},
                    step=itrn,
                )

            model.train()


    if 'xpu' in device:
        torch.xpu.synchronize()
    if 'cuda' in device:
        torch.cuda.synchronize()
    elapsed = time.time() - t0 - total_sleep

    if wandb_run is not None:
        wandb_run.summary["elapsed_active_seconds"] = elapsed
        wandb_run.finish()

    loss_per_itrn = loss_per_itrn.cpu().tolist()
    grad_norm_per_itrn = grad_norm_per_itrn.cpu().tolist()

    return {
        "lr_per_itrn": lr_per_itrn,
        "loss_per_itrn": loss_per_itrn,
        "grad_norm_per_itrn": grad_norm_per_itrn,
        "dev_losses": dev_losses,
        "dev_itrns": dev_itrns,
        "checkpoint_times": checkpoint_times,
        "elapsed": elapsed,
        "ud": ud,
        "ud_param_indices": matrix_indices,
    }