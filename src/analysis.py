"""
Plotting and evaluation utilities for the Modular Transformer models.

PLOT_BASE is the root directory; every plotting function takes a
model_name argument that creates a subfolder:
    D:\\Bar-Eden\\Act 1\\Act 1.2 Transformer\\Plots\\<model_name>\\

get_named_parameters / get_parameter_groups walk any modular model
and return human-readable names grouped by role.

All plots save automatically and close the figure (no plt.show()).
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


def _save(fig, filename, model_name=None, save_path=None):
    """Save fig and close it. Uses PLOT_BASE/<model_name>/ by default."""
    if save_path:
        path = save_path
    elif model_name:
        path = os.path.join(PLOT_BASE, model_name, filename)
    else:
        path = os.path.join(PLOT_BASE, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  NAMED PARAMETERS + GROUPING
# ═══════════════════════════════════════════════════════════════

def get_named_parameters(model):
    """
    Walk the modular model and return [(name, param)] in the same order as
    model.parameters().
    """
    named = []

    # ── embedding ──
    named.append(("embedding", model.embedding.embedding_matrix))

    # ── positional encoding ──
    if hasattr(model.positional, "positional_encoding_matrix"):
        named.append(("pos_enc", model.positional.positional_encoding_matrix))

    # ── stacked blocks ──
    if hasattr(model, "blocks"):
        for bi, block in enumerate(model.blocks):
            prefix = f"b{bi}"
            
            # Attention
            attn = block.attention
            named.append((f"{prefix}_W_Q", attn.query_weight.weight))
            named.append((f"{prefix}_W_K", attn.key_weight.weight))
            named.append((f"{prefix}_W_V", attn.value_weight.weight))
            named.append((f"{prefix}_W_out", attn.out_weight.weight))

            # Norm 1
            named.append((f"{prefix}_norm1_γ", block.norm1.gamma))
            if block.norm1.beta is not None:
                named.append((f"{prefix}_norm1_β", block.norm1.beta))

            # FFN
            ffn = block.feedforward
            if hasattr(ffn, "weight_gate"):
                named.append((f"{prefix}_ffn_gate_w", ffn.weight_gate.weight))
                if ffn.weight_gate.bias is not None:
                    named.append((f"{prefix}_ffn_gate_b", ffn.weight_gate.bias))
                    
            named.append((f"{prefix}_ffn_up_w", ffn.weight_up.weight))
            if ffn.weight_up.bias is not None:
                named.append((f"{prefix}_ffn_up_b", ffn.weight_up.bias))
                
            named.append((f"{prefix}_ffn_down_w", ffn.weight_down.weight))
            if ffn.weight_down.bias is not None:
                named.append((f"{prefix}_ffn_down_b", ffn.weight_down.bias))

            # Norm 2
            if block.norm2 is not None:
                named.append((f"{prefix}_norm2_γ", block.norm2.gamma))
                if block.norm2.beta is not None:
                    named.append((f"{prefix}_norm2_β", block.norm2.beta))

    # ── final layer norm ──
    if hasattr(model, "final_norm"):
        named.append(("ln_f_γ", model.final_norm.gamma))
        if model.final_norm.beta is not None:
            named.append(("ln_f_β", model.final_norm.beta))

    return named


def get_parameter_groups(named_params):
    """
    Group named parameters by architectural role.
    """
    groups = OrderedDict()

    for name, param in named_params:
        if name in ("embedding", "pos_enc"):
            group = "Embedding + Pos Enc"
        elif name.endswith(("_W_Q", "_W_K", "_W_V", "_W_out")) and "ffn" not in name:
            if name.startswith("b"):
                group = f"Block {name[1]} Attention"
            else:
                group = "Attention"
        elif "ffn" in name:
            if name.startswith("b"):
                group = f"Block {name[1]} FFN"
            else:
                group = "FFN"
        elif "norm" in name or "ln_f" in name:
            group = "LayerNorms"
        else:
            group = "Other"

        groups.setdefault(group, []).append((name, param))

    return groups


# ═══════════════════════════════════════════════════════════════
#  CROSS-RUN COMPARISON
# ═══════════════════════════════════════════════════════════════

def load_runs(log_path):
    """Read a JSONL log and group metric records by run_name."""
    runs = {}
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") != "metric":
                continue
            run = runs.setdefault(rec["run_name"],
                                  {"steps": [], "train_loss": [],
                                   "dev_loss": [], "lr": []})
            run["steps"].append(rec["step"])
            run["train_loss"].append(rec["train_loss"])
            run["dev_loss"].append(rec["dev_loss"])
            run["lr"].append(rec["lr"])
    return runs


def plot_run_comparison(runs, metric="dev_loss", model_name=None, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    palette = [bks.COLORS["amber"], bks.COLORS["red"],
               bks.COLORS["teal"], bks.COLORS["violet"]]

    for i, (name, run) in enumerate(runs.items()):
        color = palette[i % len(palette)]
        ax.plot(run["steps"], run[metric], marker="D", color=color, label=name)

    ax.set_xlabel("Step")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} — run comparison")
    ax.legend()

    _save(fig, "run_comparison.png", model_name, save_path)


# ═══════════════════════════════════════════════════════════════
#  SINGLE-RUN TRAINING CURVES
# ═══════════════════════════════════════════════════════════════

def plot_training_curve(results, title="Training Loss", model_name=None, save_path=None):
    loss_per_itrn = results["loss_per_itrn"]
    dev_itrns = results["dev_itrns"]
    dev_losses = results["dev_losses"]

    fig, ax = plt.subplots(figsize=(10, 5))

    steps = range(len(loss_per_itrn))
    ax.plot(steps, loss_per_itrn, color=bks.COLORS["amber"],
            alpha=0.08, linewidth=0.5)

    window = 1000
    arr = np.asarray(loss_per_itrn)
    csum = np.concatenate(([0.0], np.cumsum(arr)))
    idx = np.arange(len(arr))
    lo = np.maximum(0, idx - window)
    smoothed = (csum[idx + 1] - csum[lo]) / (idx + 1 - lo)
    ax.plot(steps, smoothed, color=bks.COLORS["amber"], linewidth=1.5,
            label="Train loss (smoothed)")

    ax.scatter(dev_itrns, dev_losses, color=bks.COLORS["red"], s=40,
               zorder=5, marker="D", label="Dev loss")
    for s, v in zip(dev_itrns, dev_losses):
        ax.text(s, v + 0.03, f"{v:.3f}", ha="center", fontsize=9,
                color=bks.COLORS["red"])

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()

    _save(fig, "training_curve.png", model_name, save_path)


def plot_lr_schedule(results, model_name=None, save_path=None):
    lr_per_itrn = results["lr_per_itrn"]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(range(len(lr_per_itrn)), lr_per_itrn,
            color=bks.COLORS["teal"], linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")

    _save(fig, "lr_schedule.png", model_name, save_path)


def plot_train_vs_dev(results, model_name=None, save_path=None):
    loss_per_itrn = results["loss_per_itrn"]
    dev_itrns = results["dev_itrns"]
    dev_losses = results["dev_losses"]

    trn_at_checkpoints = []
    for s in dev_itrns:
        w = max(0, s - 5000)
        trn_at_checkpoints.append(sum(loss_per_itrn[w:s]) / (s - w))

    fig, ax = plt.subplots(figsize=(8, 4))
    x_pos, bar_w = range(len(dev_itrns)), 0.35
    ax.bar([p - bar_w / 2 for p in x_pos], trn_at_checkpoints, bar_w,
           color=bks.COLORS["amber"], label="Train")
    ax.bar([p + bar_w / 2 for p in x_pos], dev_losses, bar_w,
           color=bks.COLORS["red"], label="Dev")
    ax.set_xticks(list(x_pos))
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
            logits.permute(0, 2, 1), data.targets,
            ignore_index=data.pad_idx)
    if name:
        print(f"{name:>5}: {loss.item():.4f}")
    return loss.item()


def evaluate_all(model, trn, dev, test):
    return {
        "train": evaluate(model, trn, "train"),
        "dev": evaluate(model, dev, "dev"),
        "test": evaluate(model, test, "test"),
    }


# ═══════════════════════════════════════════════════════════════
#  WEIGHT / ACTIVATION DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════

def weight_histogram(model, bins=50, model_name=None, save_path=None):
    all_vals = torch.cat([p.detach().flatten().cpu()
                          for p in model.parameters()])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(all_vals.numpy(), bins=bins, color=bks.COLORS["amber"])
    ax.set_title(f"Weight distribution — mean {all_vals.mean():.4f},"
                 f" std {all_vals.std():.4f}")
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")

    _save(fig, "weight_histogram.png", model_name, save_path)


def activation_saturation(model, x, bins=50, model_name=None, save_path=None):
    """
    Histogram the FFN's activations after a forward pass.
    Intercepts the lambdas returning the activation dynamically.
    """
    if not hasattr(model, "blocks"):
        print("model has no .blocks — nothing to check")
        return

    intercepted = []

    # Monkeypatch to intercept activations
    original_acts = []
    for bi, block in enumerate(model.blocks):
        orig_fn = block.feedforward.activation
        original_acts.append(orig_fn)

        def hook(x_val, fn=orig_fn, b_idx=bi):
            out = fn(x_val)
            intercepted.append((f"Block {b_idx}", out.detach().cpu().flatten()))
            return out
        
        block.feedforward.activation = hook

    # Forward pass to trigger hooks
    with torch.no_grad():
        model(x)

    # Restore originals
    for bi, block in enumerate(model.blocks):
        block.feedforward.activation = original_acts[bi]

    if not intercepted:
        return

    n = len(intercepted)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (label, out) in zip(axes, intercepted):
        suppressed = (out < -0.1).float().mean().item()
        ax.hist(out.numpy(), bins=bins, color=bks.COLORS["teal"])
        ax.set_title(f"{label} Activation — "
                     f"{suppressed * 100:.1f}% suppressed")
        ax.set_xlabel("Activation value")
        ax.set_ylabel("Count")

    fig.suptitle("FFN Activations")
    plt.tight_layout()
    _save(fig, "activation_saturation.png", model_name, save_path)


# ═══════════════════════════════════════════════════════════════
#  DIAGNOSTIC PASS (forward + backward, all models)
# ═══════════════════════════════════════════════════════════════

def diagnostic_pass(model, x, y):
    """
    One forward+backward pass retaining gradients on every stage.
    Mirrors modular_transformer.py's forward structures exactly.
    """
    for p in model.parameters():
        p.grad = None

    stages = {}

    # embedding (+ positional if present)
    emb = model.embedding(x)
    
    if hasattr(model.positional, "positional_encoding_matrix"):
        encoded = emb + model.positional()
    else:
        encoded = emb
        
    encoded.retain_grad()
    stages["encoded"] = encoded
    
    h = model.embedding_dropout(encoded)
    h.retain_grad()
    stages["post_dropout"] = h

    # Modular Blocks
    for i, block in enumerate(model.blocks):
        if block.cfg.norm_placement == "pre":
            if block.cfg.residual == "sequential":
                h_attn = h + block.dropout(block.attention(block.norm1(h)))
                h_attn.retain_grad()
                stages[f"b{i}_attn"] = h_attn
                
                h_ffn = h_attn + block.dropout(block.feedforward(block.norm2(h_attn)))
                h_ffn.retain_grad()
                stages[f"b{i}_ffn"] = h_ffn
                h = h_ffn
            elif block.cfg.residual == "parallel":
                h_par = h + block.dropout(block.attention(block.norm1(h)) + block.feedforward(block.norm1(h)))
                h_par.retain_grad()
                stages[f"b{i}_block"] = h_par
                h = h_par
                
        elif block.cfg.norm_placement == "post":
            if block.cfg.residual == "sequential":
                h_attn = block.norm1(h + block.dropout(block.attention(h)))
                h_attn.retain_grad()
                stages[f"b{i}_attn"] = h_attn
                
                h_ffn = block.norm2(h_attn + block.dropout(block.feedforward(h_attn)))
                h_ffn.retain_grad()
                stages[f"b{i}_ffn"] = h_ffn
                h = h_ffn
            elif block.cfg.residual == "parallel":
                h_par = block.norm1(h + block.dropout(block.attention(h) + block.feedforward(h)))
                h_par.retain_grad()
                stages[f"b{i}_block"] = h_par
                h = h_par

    # Final Norm & Logits
    normalized = model.final_norm(h)
    normalized.retain_grad()
    stages["normalized"] = normalized
    
    logits = normalized @ model.embedding.embedding_matrix.T
    logits.retain_grad()
    stages["logits"] = logits

    loss = torch.nn.functional.cross_entropy(
        logits.permute(0, 2, 1), y, ignore_index=-1)
    loss.backward()

    return stages, loss


def plot_activation_distributions(stages, model_name=None, save_path=None):
    names = list(stages.keys())
    fig, axes = plt.subplots(1, len(names),
                             figsize=(3.2 * len(names), 3))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        vals = stages[name].detach().cpu().flatten().numpy()
        ax.hist(vals, bins=40, color=bks.COLORS["amber"])
        ax.set_title(f"{name}\nμ={vals.mean():.3f} σ={vals.std():.3f}",
                     fontsize=9)
    fig.suptitle("Activation distributions (forward pass)")
    plt.tight_layout()

    _save(fig, "activation_distributions.png", model_name, save_path)


def plot_activation_gradients(stages, model_name=None, save_path=None):
    names = list(stages.keys())
    fig, axes = plt.subplots(1, len(names),
                             figsize=(3.2 * len(names), 3))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        vals = stages[name].grad.detach().cpu().flatten().numpy()
        ax.hist(vals, bins=40, color=bks.COLORS["red"])
        ax.set_title(f"{name} grad\n"
                     f"μ={vals.mean():.1e} σ={vals.std():.1e}",
                     fontsize=9)
    fig.suptitle("Activation gradient distributions (backward pass)")
    plt.tight_layout()

    _save(fig, "activation_gradients.png", model_name, save_path)


# ═══════════════════════════════════════════════════════════════
#  GROUPED WEIGHT GRADIENTS
# ═══════════════════════════════════════════════════════════════

def plot_weight_gradients(model, model_name=None, save_path=None):
    named = get_named_parameters(model)
    groups = get_parameter_groups(named)

    groups = OrderedDict(
        (g, [(n, p) for n, p in members if p.grad is not None])
        for g, members in groups.items()
    )
    groups = OrderedDict((g, m) for g, m in groups.items() if m)

    n_groups = len(groups)
    cols = min(n_groups, 3)
    rows = (n_groups + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(5.5 * cols, 3.5 * rows))
    if n_groups == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    palette = [bks.COLORS["amber"], bks.COLORS["red"],
               bks.COLORS["teal"], bks.COLORS["violet"],
               "#5B9BD5", "#A0856C", "#D97706", "#4FB3C9"]

    for idx, (group_name, members) in enumerate(groups.items()):
        ax = axes[idx]
        for mi, (name, param) in enumerate(members):
            vals = param.grad.detach().cpu().flatten().numpy()
            color = palette[mi % len(palette)]
            ax.hist(vals, bins=50, alpha=0.6, density=True,
                    color=color, label=name)
        ax.set_title(group_name, fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        ax.tick_params(labelsize=8)

    for idx in range(n_groups, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Weight gradient distributions (grouped)",
                 fontsize=12, y=1.02)
    plt.tight_layout()

    _save(fig, "weight_gradients.png", model_name, save_path)


# ═══════════════════════════════════════════════════════════════
#  GROUPED UPDATE-TO-DATA RATIOS
# ═══════════════════════════════════════════════════════════════

def plot_update_ratios(ud, model, model_name=None, save_path=None):
    named = get_named_parameters(model)
    groups = get_parameter_groups(named)

    params_list = list(model.parameters())
    param_to_idx = {id(p): i for i, p in enumerate(params_list)}

    def has_ud(param):
        idx = param_to_idx.get(id(param))
        return (idx is not None and param.ndim >= 2
                and idx < len(ud[0]))

    groups_filtered = OrderedDict()
    for g, members in groups.items():
        valid = [(n, p) for n, p in members if has_ud(p)]
        if valid:
            groups_filtered[g] = valid

    n_groups = len(groups_filtered)
    if n_groups == 0:
        print("No 2D parameters with UD data found.")
        return

    cols = min(n_groups, 3)
    rows = (n_groups + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(5.5 * cols, 3.5 * rows))
    if n_groups == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    palette = [bks.COLORS["amber"], bks.COLORS["red"],
               bks.COLORS["teal"], bks.COLORS["violet"],
               "#5B9BD5", "#A0856C", "#D97706", "#4FB3C9"]

    n_checkpoints = len(ud)
    x_ckpts = range(n_checkpoints)

    for idx, (group_name, members) in enumerate(groups_filtered.items()):
        ax = axes[idx]
        for mi, (name, param) in enumerate(members):
            pi = param_to_idx[id(param)]
            values = [ud[c][pi] for c in range(n_checkpoints)]
            color = palette[mi % len(palette)]
            ax.plot(x_ckpts, values, color=color, label=name,
                    linewidth=1.5, marker=".", markersize=3)

        ax.axhline(-3, color="white", linestyle="--", alpha=0.3,
                   linewidth=1, label="target (−3)")
        ax.set_title(group_name, fontsize=10)
        ax.set_xlabel("Checkpoint", fontsize=8)
        ax.set_ylabel("log₁₀(update/data)", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.tick_params(labelsize=8)

    for idx in range(n_groups, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Update-to-data ratio (grouped)", fontsize=12, y=1.02)
    plt.tight_layout()

    _save(fig, "update_ratios.png", model_name, save_path)


# ═══════════════════════════════════════════════════════════════
#  CONVENIENCE: run all diagnostics for one model
# ═══════════════════════════════════════════════════════════════

def run_all_diagnostics(model, results, x, y, model_name):
    print(f"── {model_name} ──")

    if hasattr(model, "eval"):
        model.eval()

    plot_training_curve(results, title=f"Training Loss — {model_name}",
                        model_name=model_name)
    plot_lr_schedule(results, model_name=model_name)
    plot_train_vs_dev(results, model_name=model_name)

    weight_histogram(model, model_name=model_name)
    activation_saturation(model, x, model_name=model_name)

    stages, loss = diagnostic_pass(model, x, y)
    plot_activation_distributions(stages, model_name=model_name)
    plot_activation_gradients(stages, model_name=model_name)
    plot_weight_gradients(model, model_name=model_name)

    if "ud" in results and results["ud"]:
        plot_update_ratios(results["ud"], model,
                           model_name=model_name)

    if hasattr(model, "train"):
        model.train()

    print(f"  saved to {os.path.join(PLOT_BASE, model_name)}")