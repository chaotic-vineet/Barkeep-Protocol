"""
Plotting and evaluation utilities for the modular transformer models.

PLOT_BASE is the root directory; plotting functions take a model_name
that creates a subfolder, or an explicit save_path.

Three families:
    single-run   — training curve, lr schedule, train-vs-dev, weight and
                   activation diagnostics (need a live model / results dict)
    cross-run    — load_log + plot_run_comparison (need only a JSONL)
    ablation     — the Act 1.3 study plots + analyze_ablation driver
                   (need only a JSONL)

All plots save automatically and close the figure (no plt.show()).
Colors come from barkeep_style.ACCENT_CYCLE — no local palettes.
"""

import json
import os
from collections import OrderedDict

import numpy as np
import torch
import matplotlib.pyplot as plt

import src.barkeep_style as bks

bks.apply_style()

PLOT_BASE = r"D:\Bar-Eden\Act 1\Act 1.2 Transformer\Plots"
PALETTE = bks.ACCENT_CYCLE


def _save(fig, filename, model_name=None, save_path=None):
    """Save fig and close it. Uses PLOT_BASE/<model_name>/ by default."""
    path = save_path or os.path.join(PLOT_BASE, model_name or "", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _grid(n, cols=3, w=5.5, h=3.5, **kwargs):
    """Subplot grid for n panels; returns (fig, flat axes) with extras hidden."""
    cols = min(n, cols)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(w * cols, h * rows), **kwargs)
    axes = np.array(axes).flatten() if n > 1 else np.array([axes])
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes


# ═══════════════════════════════════════════════════════════════
#  NAMED PARAMETERS + GROUPING
# ═══════════════════════════════════════════════════════════════

def get_named_parameters(model):
    """Walk the model; return [(name, param)] in model.parameters() order."""
    named = [("embedding", model.embedding.embedding_matrix)]

    if model.positional.parameters():
        named.append(("pos_enc", model.positional.positional_encoding_matrix))

    def add_norm(prefix, norm):
        named.append((f"{prefix}_γ", norm.gamma))
        if norm.beta is not None:
            named.append((f"{prefix}_β", norm.beta))

    def add_linear(prefix, linear):
        named.append((f"{prefix}_w", linear.weight))
        if linear.bias is not None:
            named.append((f"{prefix}_b", linear.bias))

    for bi, block in enumerate(getattr(model, "blocks", [])):
        p = f"b{bi}"
        attn = block.attention
        named += [(f"{p}_W_Q", attn.query_weight.weight),
                  (f"{p}_W_K", attn.key_weight.weight),
                  (f"{p}_W_V", attn.value_weight.weight),
                  (f"{p}_W_out", attn.out_weight.weight)]
        # FFN order mirrors FeedForwardNetwork.parameters(): up, gate, down
        ffn = block.feedforward
        add_linear(f"{p}_ffn_up", ffn.weight_up)
        if hasattr(ffn, "weight_gate"):
            add_linear(f"{p}_ffn_gate", ffn.weight_gate)
        add_linear(f"{p}_ffn_down", ffn.weight_down)
        add_norm(f"{p}_norm1", block.norm1)
        if block.norm2 is not None:
            add_norm(f"{p}_norm2", block.norm2)

    add_norm("ln_f", model.final_norm)
    return named


def get_parameter_groups(named_params):
    """Group named parameters by architectural role."""
    groups = OrderedDict()
    for name, param in named_params:
        if name in ("embedding", "pos_enc"):
            group = "Embedding + Pos Enc"
        elif "_W_" in name:
            group = f"Block {name[1]} Attention"
        elif "ffn" in name:
            group = f"Block {name[1]} FFN"
        else:
            group = "Norms"
        groups.setdefault(group, []).append((name, param))
    return groups


# ═══════════════════════════════════════════════════════════════
#  JSONL LOADING + CROSS-RUN COMPARISON
# ═══════════════════════════════════════════════════════════════

RUN_ORDER = ["baseline", "modern_full", "modern_minus_rope",
             "modern_minus_gqa", "modern_minus_swiglu", "modern_minus_rms"]

AXIS_LABELS = {
    "modern_minus_rope":   "RoPE",
    "modern_minus_gqa":    "GQA",
    "modern_minus_swiglu": "SwiGLU",
    "modern_minus_rms":    "RMSNorm",
}


def load_log(log_path):
    """
    Read a JSONL log and return
        {"runs":    {run_name: {steps, train_loss, dev_loss, lr, grad_norm}},
         "configs": {run_name: full config record},
         "summary": rows or None}
    Runs follow RUN_ORDER where applicable, insertion order otherwise.
    """
    runs, configs, summary = {}, {}, None
    with open(log_path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            kind = rec.get("type")
            if kind == "config":
                configs[rec["run_name"]] = rec
            elif kind == "metric":
                run = runs.setdefault(rec["run_name"], {
                    "steps": [], "train_loss": [], "dev_loss": [],
                    "lr": [], "grad_norm": []})
                run["steps"].append(rec["step"])
                for key in ("train_loss", "dev_loss", "lr", "grad_norm"):
                    run[key].append(rec.get(key))
            elif kind == "summary":
                summary = rec["rows"]

    ordered = OrderedDict((n, runs.pop(n)) for n in RUN_ORDER if n in runs)
    ordered.update(runs)
    return {"runs": ordered, "configs": configs, "summary": summary}


def load_runs(log_path):
    """Back-compat wrapper: just the metric curves, keyed by run_name."""
    return load_log(log_path)["runs"]


def plot_run_comparison(runs, metric="dev_loss", model_name=None, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, run) in enumerate(runs.items()):
        ax.plot(run["steps"], run[metric], marker="D", markersize=4,
                color=PALETTE[i % len(PALETTE)], label=name)
    ax.set_xlabel("Step")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} — run comparison")
    ax.legend(fontsize=9)
    _save(fig, f"run_comparison_{metric}.png", model_name, save_path)


# ═══════════════════════════════════════════════════════════════
#  SINGLE-RUN TRAINING CURVES (need the results dict)
# ═══════════════════════════════════════════════════════════════

def plot_training_curve(results, title="Training Loss", model_name=None, save_path=None):
    arr = np.asarray(results["loss_per_itrn"])
    steps = np.arange(len(arr))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, arr, color=bks.COLORS["amber"], alpha=0.08, linewidth=0.5)

    window = 1000
    csum = np.concatenate(([0.0], np.cumsum(arr)))
    lo = np.maximum(0, steps - window)
    ax.plot(steps, (csum[steps + 1] - csum[lo]) / (steps + 1 - lo),
            color=bks.COLORS["amber"], linewidth=1.5, label="Train loss (smoothed)")

    ax.scatter(results["dev_itrns"], results["dev_losses"],
               color=bks.COLORS["red"], s=40, zorder=5, marker="D", label="Dev loss")
    for s, v in zip(results["dev_itrns"], results["dev_losses"]):
        ax.text(s, v + 0.03, f"{v:.3f}", ha="center", fontsize=9,
                color=bks.COLORS["red"])

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    _save(fig, "training_curve.png", model_name, save_path)


def plot_lr_schedule(results, model_name=None, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(results["lr_per_itrn"], color=bks.COLORS["teal"], linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    _save(fig, "lr_schedule.png", model_name, save_path)


def plot_train_vs_dev(results, model_name=None, save_path=None):
    loss, dev_itrns = results["loss_per_itrn"], results["dev_itrns"]
    trn = [sum(loss[max(0, s - 5000):s]) / (s - max(0, s - 5000)) for s in dev_itrns]

    fig, ax = plt.subplots(figsize=(8, 4))
    x, w = np.arange(len(dev_itrns)), 0.35
    ax.bar(x - w / 2, trn, w, color=bks.COLORS["amber"], label="Train")
    ax.bar(x + w / 2, results["dev_losses"], w, color=bks.COLORS["red"], label="Dev")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s // 1000}K" for s in dev_itrns])
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Train vs Dev Loss at Checkpoints")
    ax.legend()
    _save(fig, "train_vs_dev.png", model_name, save_path)


# ═══════════════════════════════════════════════════════════════
#  EVALUATION
# ═══════════════════════════════════════════════════════════════

def evaluate(model, data, name=""):
    with torch.no_grad():
        logits = model(data.inputs)
        loss = torch.nn.functional.cross_entropy(
            logits.permute(0, 2, 1), data.targets, ignore_index=data.pad_idx)
    if name:
        print(f"{name:>5}: {loss.item():.4f}")
    return loss.item()


def evaluate_all(model, trn, dev, test):
    return {"train": evaluate(model, trn, "train"),
            "dev": evaluate(model, dev, "dev"),
            "test": evaluate(model, test, "test")}


# ═══════════════════════════════════════════════════════════════
#  WEIGHT / ACTIVATION DIAGNOSTICS (need a live model)
# ═══════════════════════════════════════════════════════════════

def weight_histogram(model, bins=50, model_name=None, save_path=None):
    vals = torch.cat([p.detach().flatten().cpu() for p in model.parameters()])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(vals.numpy(), bins=bins, color=bks.COLORS["amber"])
    ax.set_title(f"Weight distribution — mean {vals.mean():.4f}, std {vals.std():.4f}")
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    _save(fig, "weight_histogram.png", model_name, save_path)


def activation_saturation(model, x, bins=50, model_name=None, save_path=None):
    """Histogram each block's FFN activation output on one forward pass."""
    if not hasattr(model, "blocks"):
        print("model has no .blocks — nothing to check")
        return

    intercepted, originals = [], []
    for bi, block in enumerate(model.blocks):
        orig = block.feedforward.activation
        originals.append(orig)

        def hook(v, fn=orig, b=bi):
            out = fn(v)
            intercepted.append((f"Block {b}", out.detach().cpu().flatten()))
            return out
        block.feedforward.activation = hook

    with torch.no_grad():
        model(x)
    for block, orig in zip(model.blocks, originals):
        block.feedforward.activation = orig

    fig, axes = _grid(len(intercepted), cols=len(intercepted), w=6, h=4)
    for ax, (label, out) in zip(axes, intercepted):
        suppressed = (out < -0.1).float().mean().item()
        ax.hist(out.numpy(), bins=bins, color=bks.COLORS["teal"])
        ax.set_title(f"{label} Activation — {suppressed * 100:.1f}% suppressed")
        ax.set_xlabel("Activation value")
        ax.set_ylabel("Count")
    fig.suptitle("FFN Activations")
    plt.tight_layout()
    _save(fig, "activation_saturation.png", model_name, save_path)


def diagnostic_pass(model, x, y):
    """
    One forward+backward pass retaining gradients on every stage.
    Mirrors modular_transformer.py's forward structure exactly.
    """
    for p in model.parameters():
        p.grad = None

    stages = {}

    def keep(name, tensor):
        tensor.retain_grad()
        stages[name] = tensor
        return tensor

    emb = model.embedding(x)
    # Learned/Sinusoidal add at the embedding; RoPE/ALiBi act inside attention
    if hasattr(model.positional, "positional_encoding_matrix"):
        emb = emb + model.positional()
    keep("encoded", emb)
    h = keep("post_dropout", model.embedding_dropout(emb))

    for i, block in enumerate(model.blocks):
        pre, seq = block.cfg.norm_placement == "pre", block.cfg.residual == "sequential"
        if pre and seq:
            h = keep(f"b{i}_attn", h + block.dropout(block.attention(block.norm1(h))))
            h = keep(f"b{i}_ffn", h + block.dropout(block.feedforward(block.norm2(h))))
        elif pre:
            h = keep(f"b{i}_block", h + block.dropout(
                block.attention(block.norm1(h)) + block.feedforward(block.norm1(h))))
        elif seq:
            h = keep(f"b{i}_attn", block.norm1(h + block.dropout(block.attention(h))))
            h = keep(f"b{i}_ffn", block.norm2(h + block.dropout(block.feedforward(h))))
        else:
            h = keep(f"b{i}_block", block.norm1(
                h + block.dropout(block.attention(h) + block.feedforward(h))))

    normalized = keep("normalized", model.final_norm(h))
    logits = keep("logits", normalized @ model.embedding.embedding_matrix.T)

    loss = torch.nn.functional.cross_entropy(logits.permute(0, 2, 1), y, ignore_index=-1)
    loss.backward()
    return stages, loss


def _stage_hist_grid(stages, values_of, color, suptitle, fname,
                     model_name=None, save_path=None):
    names = list(stages.keys())
    fig, axes = _grid(len(names), cols=len(names), w=3.2, h=3)
    for ax, name in zip(axes, names):
        vals = values_of(stages[name]).detach().cpu().flatten().numpy()
        ax.hist(vals, bins=40, color=color)
        ax.set_title(f"{name}\nμ={vals.mean():.3g} σ={vals.std():.3g}", fontsize=9)
    fig.suptitle(suptitle)
    plt.tight_layout()
    _save(fig, fname, model_name, save_path)


def plot_activation_distributions(stages, model_name=None, save_path=None):
    _stage_hist_grid(stages, lambda t: t, bks.COLORS["amber"],
                     "Activation distributions (forward pass)",
                     "activation_distributions.png", model_name, save_path)


def plot_activation_gradients(stages, model_name=None, save_path=None):
    _stage_hist_grid(stages, lambda t: t.grad, bks.COLORS["red"],
                     "Activation gradient distributions (backward pass)",
                     "activation_gradients.png", model_name, save_path)


def plot_weight_gradients(model, model_name=None, save_path=None):
    groups = OrderedDict(
        (g, m) for g, m in (
            (g, [(n, p) for n, p in members if p.grad is not None])
            for g, members in get_parameter_groups(get_named_parameters(model)).items()
        ) if m
    )
    fig, axes = _grid(len(groups))
    for ax, (group_name, members) in zip(axes, groups.items()):
        for mi, (name, param) in enumerate(members):
            ax.hist(param.grad.detach().cpu().flatten().numpy(), bins=50,
                    alpha=0.6, density=True,
                    color=PALETTE[mi % len(PALETTE)], label=name)
        ax.set_title(group_name, fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        ax.tick_params(labelsize=8)
    fig.suptitle("Weight gradient distributions (grouped)", fontsize=12, y=1.02)
    plt.tight_layout()
    _save(fig, "weight_gradients.png", model_name, save_path)


def plot_update_ratios(results, model, model_name=None, save_path=None):
    """
    log10(update/data) per checkpoint, grouped by role.

    ud columns are indexed by POSITION within ud_param_indices, not by
    raw parameter index — results["ud_param_indices"] provides the map.
    """
    ud, ud_indices = results["ud"], results["ud_param_indices"]
    col_of = {param_idx: col for col, param_idx in enumerate(ud_indices)}
    param_to_idx = {id(p): i for i, p in enumerate(model.parameters())}

    groups = OrderedDict(
        (g, m) for g, m in (
            (g, [(n, p) for n, p in members
                 if param_to_idx.get(id(p)) in col_of])
            for g, members in get_parameter_groups(get_named_parameters(model)).items()
        ) if m
    )
    if not groups:
        print("No matrix parameters with UD data found.")
        return

    fig, axes = _grid(len(groups))
    x = range(len(ud))
    for ax, (group_name, members) in zip(axes, groups.items()):
        for mi, (name, param) in enumerate(members):
            col = col_of[param_to_idx[id(param)]]
            ax.plot(x, [ud[c][col] for c in x], color=PALETTE[mi % len(PALETTE)],
                    label=name, linewidth=1.5, marker=".", markersize=3)
        ax.axhline(-3, color=bks.COLORS["text_mid"], linestyle="--",
                   alpha=0.5, linewidth=1, label="target (−3)")
        ax.set_title(group_name, fontsize=10)
        ax.set_xlabel("Checkpoint", fontsize=8)
        ax.set_ylabel("log₁₀(update/data)", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.tick_params(labelsize=8)
    fig.suptitle("Update-to-data ratio (grouped)", fontsize=12, y=1.02)
    plt.tight_layout()
    _save(fig, "update_ratios.png", model_name, save_path)


def run_all_diagnostics(model, results, x, y, model_name, out_dir=None):
    """Every single-run plot for one model. out_dir overrides PLOT_BASE."""
    print(f"── {model_name} ──")
    base = os.path.join(out_dir or PLOT_BASE, model_name)
    path = lambda f: os.path.join(base, f)

    model.eval()
    plot_training_curve(results, title=f"Training Loss — {model_name}",
                        save_path=path("training_curve.png"))
    plot_lr_schedule(results, save_path=path("lr_schedule.png"))
    plot_train_vs_dev(results, save_path=path("train_vs_dev.png"))
    weight_histogram(model, save_path=path("weight_histogram.png"))
    activation_saturation(model, x, save_path=path("activation_saturation.png"))

    stages, _ = diagnostic_pass(model, x, y)
    plot_activation_distributions(stages, save_path=path("activation_distributions.png"))
    plot_activation_gradients(stages, save_path=path("activation_gradients.png"))
    plot_weight_gradients(model, save_path=path("weight_gradients.png"))
    if results.get("ud"):
        plot_update_ratios(results, model, save_path=path("update_ratios.png"))

    model.train()
    print(f"  saved to {base}")


# ═══════════════════════════════════════════════════════════════
#  ABLATION v2 — seed-aware aggregation
#
#  v2 runs are named {config}_s{seed}. These utilities group runs by
#  base config, aggregate over seeds (mean + min–max band), and put
#  the measured noise floor next to every effect size.
#
#  dev_full (whole-dev-split loss) comes from the per-run results
#  JSONs in <results_dir>; if absent, falls back to the final
#  sampled-batch dev loss from the JSONL (noisier — flagged in the
#  table header).
# ═══════════════════════════════════════════════════════════════

def _split_seed(run_name):
    base, _, seed = run_name.rpartition("_s")
    return (base, int(seed)) if base and seed.isdigit() else (run_name, None)


def load_ablation_v2(log_path, results_dir=None):
    """
    Group a v2 JSONL by base config. Returns
        {"configs_grouped": {base: {"seeds": {seed: run_curves},
                                    "dev_full": {seed: float} | {},
                                    "num_params": int}},
         "used_dev_full": bool}
    Curve dict per seed is the load_log metric structure.
    """
    log = load_log(log_path)
    grouped = OrderedDict()

    for run_name, curves in log["runs"].items():
        base, seed = _split_seed(run_name)
        g = grouped.setdefault(base, {"seeds": OrderedDict(), "dev_full": {},
                                      "num_params": None})
        g["seeds"][seed] = curves
        cfg = log["configs"].get(run_name, {})
        g["num_params"] = cfg.get("num_params", g["num_params"])

        if results_dir is not None:
            path = os.path.join(results_dir, f"{run_name}.json")
            if os.path.exists(path):
                with open(path) as f:
                    g["dev_full"][seed] = json.load(f)["dev_full"]

    used_dev_full = all(g["dev_full"] for g in grouped.values())
    if not used_dev_full:
        for g in grouped.values():  # fallback: final sampled dev loss
            g["dev_full"] = {s: c["dev_loss"][-1] for s, c in g["seeds"].items()}

    order = [n for n in RUN_ORDER if n in grouped]
    ordered = OrderedDict((n, grouped.pop(n)) for n in order)
    ordered.update(grouped)
    return {"configs_grouped": ordered, "used_dev_full": used_dev_full}


def _stats(vals):
    m = sum(vals) / len(vals)
    return m, min(vals), max(vals), max(vals) - min(vals)


def print_ablation_v2_table(abl2, reference="modern_full"):
    grouped = abl2["configs_grouped"]
    metric = "full-dev loss" if abl2["used_dev_full"] else \
             "final SAMPLED dev loss (dev_full unavailable — noisier)"
    ref_mean = _stats(list(grouped[reference]["dev_full"].values()))[0] \
        if reference in grouped else None

    print(f"metric: {metric}")
    header = (f"{'config':<22}{'params':>12}{'mean':>10}{'min':>9}"
              f"{'max':>9}{'spread':>9}{'n':>4}{'Δ vs ref':>10}")
    print(header)
    print("-" * len(header))
    rows = {}
    for name, g in grouped.items():
        m, lo, hi, spread = _stats(list(g["dev_full"].values()))
        rows[name] = (m, lo, hi, spread)
        d = "ref" if name == reference else (
            f"{m - ref_mean:+.4f}" if ref_mean is not None else "")
        print(f"{name:<22}{g['num_params']:>12,}{m:>10.4f}{lo:>9.4f}"
              f"{hi:>9.4f}{spread:>9.4f}{len(g['dev_full']):>4}{d:>10}")

    # non-additivity check on means, with the noise floor beside it
    strips = [n for n in grouped if n in AXIS_LABELS]
    if strips and reference in rows and "baseline" in rows:
        delta_sum = sum(rows[n][0] - rows[reference][0] for n in strips)
        gap = rows["baseline"][0] - rows[reference][0]
        max_spread = max(r[3] for r in rows.values())
        print(f"\nsum of strip deltas (means): {delta_sum:+.4f} | "
              f"baseline−{reference} gap: {gap:+.4f} | "
              f"discrepancy: {delta_sum - gap:+.4f} | "
              f"largest per-config spread: {max_spread:.4f}")
    return rows


def plot_ablation_v2_dev_bands(abl2, model_name=None, save_path=None):
    """Per config: mean dev curve across seeds with a min–max band."""
    grouped = abl2["configs_grouped"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, g) in enumerate(grouped.items()):
        color = PALETTE[i % len(PALETTE)]
        seed_curves = list(g["seeds"].values())
        steps = seed_curves[0]["steps"]
        assert all(c["steps"] == steps for c in seed_curves), f"{name}: step mismatch"
        per_step = list(zip(*[c["dev_loss"] for c in seed_curves]))
        mean = [sum(v) / len(v) for v in per_step]
        style = dict(linewidth=2.2) if name in ("baseline", "modern_full") \
            else dict(linewidth=1.4, linestyle="--")
        ax.plot(steps, mean, marker="D", markersize=3.5, color=color,
                label=f"{name} (n={len(seed_curves)})", **style)
        ax.fill_between(steps, [min(v) for v in per_step],
                        [max(v) for v in per_step], color=color, alpha=0.15)
    ax.set_xlabel("Step")
    ax.set_ylabel("Dev loss (sampled batch)")
    ax.set_title("Dev loss — mean over seeds, min–max band")
    ax.legend(fontsize=9)
    _save(fig, "ablation_v2_dev_bands.png", model_name, save_path)


def plot_ablation_v2_seed_clouds(abl2, model_name=None, save_path=None):
    """
    Every seed's headline loss as a point, per config — effect vs noise
    read directly: disjoint clouds are real effects, interleaved clouds
    are inside the noise floor.
    """
    grouped = abl2["configs_grouped"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (name, g) in enumerate(grouped.items()):
        color = PALETTE[i % len(PALETTE)]
        vals = list(g["dev_full"].values())
        ax.scatter([i] * len(vals), vals, color=color, s=55, marker="D", zorder=5)
        m = sum(vals) / len(vals)
        ax.hlines(m, i - 0.22, i + 0.22, color=color, linewidth=2)
    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels(grouped.keys(), rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Full-dev loss" if abl2["used_dev_full"] else "Final sampled dev loss")
    ax.set_title("Per-seed results — clouds vs noise floor (bar = mean)")
    _save(fig, "ablation_v2_seed_clouds.png", model_name, save_path)


def plot_ablation_v2_strip_deltas(abl2, reference="modern_full",
                                  model_name=None, save_path=None):
    """
    Mean strip penalty per axis with whiskers spanning the min–max of
    per-seed deltas (each seed's value minus the reference MEAN).
    Whiskers crossing zero = effect inside the measured noise.
    """
    grouped = abl2["configs_grouped"]
    ref_mean = _stats(list(grouped[reference]["dev_full"].values()))[0]
    ref_params = grouped[reference]["num_params"]

    names = [n for n in grouped if n in AXIS_LABELS]
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, name in enumerate(names):
        vals = [v - ref_mean for v in grouped[name]["dev_full"].values()]
        m, lo, hi, _ = _stats(vals)
        color = PALETTE[i % len(PALETTE)]
        ax.bar(i, m, color=color, width=0.6)
        ax.vlines(i, lo, hi, color=bks.COLORS["text_bright"], linewidth=1.5)
        dp = grouped[name]["num_params"] - ref_params
        ax.text(i, hi + 0.004, f"{m:+.4f}\n(Δ params {dp:+,})",
                ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color=bks.COLORS["text_mid"], linewidth=1, alpha=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([AXIS_LABELS[n] for n in names])
    ax.set_ylabel(f"Loss − {reference} (mean, min–max whiskers)")
    ax.set_title("Leave-one-out penalties with measured noise\n"
                 "(whisker crossing zero = inside the noise floor)")
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - 0.01, ymax + 0.03)
    _save(fig, "ablation_v2_strip_deltas.png", model_name, save_path)


def analyze_ablation_v2(log_path, results_dir=None, out_dir=None,
                        reference="modern_full"):
    """
    The v2 driver: seed-grouped table + the three cross-run figures.
    results_dir should hold the per-run {run}.json files (for dev_full).
    """
    abl2 = load_ablation_v2(log_path, results_dir)
    print_ablation_v2_table(abl2, reference=reference)

    base = out_dir or os.path.join(PLOT_BASE, "ablation_1_3_v2")
    path = lambda f: os.path.join(base, f)
    plot_ablation_v2_dev_bands(abl2, save_path=path("ablation_v2_dev_bands.png"))
    plot_ablation_v2_seed_clouds(abl2, save_path=path("ablation_v2_seed_clouds.png"))
    plot_ablation_v2_strip_deltas(abl2, reference,
                                  save_path=path("ablation_v2_strip_deltas.png"))
    print(f"\nplots saved to {base}")
    return abl2