"""
DISCO Attacker Overlap Analysis: Show that attackers don't target DISCO's directions.

Simulates a fine-tuning attacker (SGD on forget data), then measures:
1. What fraction of attacker's ΔW energy lands on DISCO's top-m directions
2. What fraction lands on PCA's top-m directions
3. Comparison: attacker energy on DISCO vs PCA vs random directions

This directly proves robustness without defining "attacker subspace" a priori.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/disco_attacker_overlap.py \
        --config-name=unlearn.yaml \
        experiment=unlearn/wmdp_low_mi/default \
        model=Qwen3-8B-Base \
        task_name=DISCO_ATTACKER_QWEN3_8B_BIO
"""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import hydra
import torch as pt
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from data import get_collators, get_data
from model import get_model
from trainer.utils import seed_everything
from interp_utils import (
    PLOT_DIR, collect_activations, load_wikitext,
    select_layers, solve_generalised_eigenvalue, save_json,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAVE_DIR = Path(__file__).resolve().parent.parent / "saves" / "disco_motivation"


def simulate_attacker(model, forget_dataset, collator, n_steps=50, lr=1e-4):
    """Simulate fine-tuning attacker: SGD on forget data for n_steps.
    Returns dict of weight deltas per module name."""
    # Save original weights
    original_state = {k: v.clone() for k, v in model.state_dict().items()
                      if "mlp" in k and "weight" in k}

    # Enable grad on MLP weights
    for name, param in model.named_parameters():
        param.requires_grad = "mlp" in name and "weight" in name

    optimizer = pt.optim.SGD(
        [p for n, p in model.named_parameters() if p.requires_grad],
        lr=lr
    )

    loader = DataLoader(forget_dataset, batch_size=8, collate_fn=collator, shuffle=True)
    model.train()

    step = 0
    while step < n_steps:
        for batch in loader:
            if step >= n_steps:
                break
            inputs = {
                "input_ids": batch["input_ids"].to(model.device),
                "attention_mask": batch["attention_mask"].to(model.device),
                "labels": batch["labels"].to(model.device),
            }
            loss = model(**inputs).loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

    # Compute deltas
    deltas = {}
    current_state = model.state_dict()
    for key in original_state:
        deltas[key] = (current_state[key].float().cpu() - original_state[key].float().cpu())

    # Restore original weights
    model.load_state_dict({k: v for k, v in zip(model.state_dict().keys(),
                          [original_state.get(k, model.state_dict()[k]) for k in model.state_dict().keys()])},
                         strict=False)
    # Simpler restore
    for key, val in original_state.items():
        parts = key.split(".")
        module = model
        for p in parts[:-1]:
            module = getattr(module, p)
        setattr(module, parts[-1], pt.nn.Parameter(val))

    model.eval()
    return deltas


def compute_energy_fraction(delta_W, directions, m):
    """Fraction of ||ΔW||_F^2 that projects onto the first m directions.
    directions: (D_in, n_dirs) — projects along input (row) dimension of ΔW."""
    dirs = directions[:, :m].float()
    # Project each row of ΔW onto directions
    proj = delta_W.float() @ dirs  # (D_out, m)
    energy_in_dirs = (proj ** 2).sum()
    total_energy = (delta_W.float() ** 2).sum()
    return (energy_in_dirs / total_energy).item() if total_energy > 0 else 0.0


def plot_attacker_overlap(results, save_path, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    layers = sorted(results.keys())
    x = np.arange(len(layers))
    w = 0.25

    disco_vals = [results[l]["disco"] for l in layers]
    pca_vals = [results[l]["pca"] for l in layers]
    random_vals = [results[l]["random"] for l in layers]

    ax.bar(x - w, pca_vals, w, label="PCA top-m", color="#e74c3c", alpha=0.8)
    ax.bar(x, disco_vals, w, label="DISCO top-m", color="#2ecc71", alpha=0.8)
    ax.bar(x + w, random_vals, w, label="Random m dirs", color="#95a5a6", alpha=0.8)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Fraction of attacker ΔW energy")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="unlearn.yaml")
def main(cfg: DictConfig):
    seed_everything(cfg.trainer.args.seed)
    task_name = cfg.get("task_name", "DISCO_ATTACKER")

    model, tokenizer = get_model(cfg.model)
    model = model.cuda().eval()

    data = get_data(cfg.data, mode="unlearn", tokenizer=tokenizer, template_args=cfg.model.template_args)
    collator = get_collators(cfg.collator, tokenizer=tokenizer)
    train_data = data["train"]

    layers = select_layers(len(model.model.layers))
    m = 200  # number of directions to compare

    # --- Step 1: Collect activations and compute DISCO + PCA directions ---
    print("Collecting activations and computing directions...")
    directions_per_layer = {}

    for layer_idx in layers:
        target = model.model.layers[layer_idx].mlp.gate_proj
        print(f"\n  Layer {layer_idx}: collecting activations...")

        f_acts = collect_activations(model, train_data.forget, collator, target).float()
        r_acts = collect_activations(model, train_data.retain, collator, target).float()

        mean_f = f_acts.mean(dim=0)
        Sigma_f = ((f_acts - mean_f).T @ (f_acts - mean_f)) / (f_acts.shape[0] - 1)
        mean_r = r_acts.mean(dim=0)
        Sigma_r = ((r_acts - mean_r).T @ (r_acts - mean_r)) / (r_acts.shape[0] - 1)

        # DISCO directions
        gen_evals, gen_evecs = solve_generalised_eigenvalue(Sigma_f.cuda(), Sigma_r.cuda())
        gen_evecs = gen_evecs.cpu()

        # PCA directions
        _, _, V_pca = pt.svd_lowrank(Sigma_f.cuda(), q=min(m, Sigma_f.shape[0]))
        V_pca = V_pca.cpu()

        # Random directions (for baseline)
        random_dirs = pt.randn(Sigma_f.shape[0], m)
        random_dirs = random_dirs / random_dirs.norm(dim=0, keepdim=True)

        directions_per_layer[layer_idx] = {
            "disco": gen_evecs,
            "pca": V_pca,
            "random": random_dirs,
            "disco_lambdas": gen_evals.cpu(),
        }

        del f_acts, r_acts, Sigma_f, Sigma_r
        pt.cuda.empty_cache()

    # --- Step 2: Simulate attacker ---
    print("\nSimulating fine-tuning attacker (50 SGD steps)...")
    deltas = simulate_attacker(model, train_data.forget, collator, n_steps=50, lr=1e-4)

    # --- Step 3: Measure overlap ---
    print("\nMeasuring attacker energy overlap:")
    results = {}

    for layer_idx in layers:
        gate_key = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
        if gate_key not in deltas:
            print(f"  Layer {layer_idx}: no delta found, skipping")
            continue

        delta_W = deltas[gate_key]
        dirs = directions_per_layer[layer_idx]

        disco_frac = compute_energy_fraction(delta_W, dirs["disco"], m)
        pca_frac = compute_energy_fraction(delta_W, dirs["pca"], m)
        random_frac = compute_energy_fraction(delta_W, dirs["random"], m)

        results[layer_idx] = {
            "disco": disco_frac,
            "pca": pca_frac,
            "random": random_frac,
        }

        print(f"  Layer {layer_idx}: PCA={pca_frac:.1%}  DISCO={disco_frac:.1%}  "
              f"Random={random_frac:.1%}  (PCA/DISCO={pca_frac/max(disco_frac,1e-10):.1f}x)")

    # --- Step 4: Plot and save ---
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    plot_attacker_overlap(results,
        PLOT_DIR / f"disco_attacker_overlap_{task_name}.pdf",
        f"Attacker ΔW energy on DISCO vs PCA directions (top-{m})")

    save_json(results, SAVE_DIR / f"disco_attacker_overlap_{task_name}.json")


if __name__ == "__main__":
    main()
