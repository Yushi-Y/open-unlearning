"""
DISCO Motivation Analysis: Empirical evidence for discriminative collapse.

Forward-only — no training, no backward passes.

Per layer, computes:
1. Generalised eigenvalue spectrum (λ = forget_var / retain_var)
2. Tiered variance ratios for direction tiers sorted by λ
3. PCA vs DISCO direction comparison
4. Token projection of high-λ and low-λ directions through lm_head

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/disco_motivation.py \
        --config-name=unlearn.yaml \
        experiment=unlearn/wmdp_low_mi/default \
        model=Qwen3-8B-Base \
        task_name=DISCO_MOTIVATION_QWEN3_8B_BIO
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import hydra
import torch as pt
import numpy as np
from omegaconf import DictConfig

from data import get_collators, get_data
from model import get_model
from trainer.utils import seed_everything
from interp_utils import (
    SAVE_DIR, PLOT_DIR, collect_activations, load_wikitext,
    select_layers, solve_generalised_eigenvalue, compute_tiered_variance,
    save_json,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DISCO_DIR = Path(__file__).resolve().parent.parent / "saves" / "disco_motivation"
TIERS = {"0-3": (0, 3), "3-10": (3, 10), "10-30": (10, 30),
         "30-100": (30, 100), "100-300": (100, 300), "300+": (300, 10000)}


def plot_eigenvalue_spectrum(eigenvalues, save_path, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    vals = eigenvalues.cpu().numpy()
    ax1.plot(range(min(500, len(vals))), vals[:500], color="#e74c3c", linewidth=1.2)
    ax1.set_xlabel("Direction index (sorted by λ)")
    ax1.set_ylabel("λ = forget_var / retain_var")
    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="λ=1")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.semilogy(range(len(vals)), vals, color="#e74c3c", linewidth=1.2)
    ax2.set_xlabel("Direction index"); ax2.set_ylabel("λ (log)")
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5); ax2.grid(alpha=0.3)
    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout(); fig.savefig(save_path); plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_tiered_ratios(tier_data, save_path, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(tier_data.keys())
    x = np.arange(len(names)); w = 0.3
    ax.bar(x - w/2, [tier_data[n]["forget"] for n in names], w, label="Forget", color="#e74c3c", alpha=0.8)
    ax.bar(x + w/2, [tier_data[n]["retain"] for n in names], w, label="Retain", color="#3498db", alpha=0.8)
    ax2 = ax.twinx()
    ax2.plot(x, [tier_data[n]["ratio"] for n in names], "ko-", linewidth=2, markersize=8, label="Ratio")
    ax2.set_ylabel("Forget / Retain ratio"); ax2.legend(loc="upper right")
    ax.set_xlabel("Direction tier (by λ)"); ax.set_ylabel("Variance fraction")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
    ax.legend(loc="upper left"); ax.grid(alpha=0.3, axis="y")
    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout(); fig.savefig(save_path); plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_pca_vs_disco(pca_lambdas, disco_lambdas, save_path, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    n = min(50, len(pca_lambdas))
    x = np.arange(n)
    ax.plot(x, disco_lambdas[:n].cpu().numpy(), "o-", color="#2ecc71", label="DISCO", markersize=3)
    ax.plot(x, pca_lambdas[:n].cpu().numpy(), "s-", color="#e74c3c", label="PCA", markersize=3)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Direction index (top-50)"); ax.set_ylabel("λ")
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); fig.savefig(save_path); plt.close(fig)
    print(f"  Saved: {save_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="unlearn.yaml")
def main(cfg: DictConfig):
    seed_everything(cfg.trainer.args.seed)
    task_name = cfg.get("task_name", "DISCO_MOTIVATION")

    # Load model (no trainer)
    model, tokenizer = get_model(cfg.model)
    model = model.cuda().eval()

    # Load data
    data = get_data(cfg.data, mode="unlearn", tokenizer=tokenizer, template_args=cfg.model.template_args)
    collator = get_collators(cfg.collator, tokenizer=tokenizer)
    train_data = data["train"]
    wikitext_ds = load_wikitext(tokenizer)

    layers = select_layers(len(model.model.layers))
    DISCO_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for layer_idx in layers:
        target = model.model.layers[layer_idx].mlp.gate_proj
        print(f"\n{'='*60}\nLayer {layer_idx}/{len(model.model.layers)-1}\n{'='*60}")

        # Collect activations (forward only)
        print("  Collecting forget activations...")
        f_acts = collect_activations(model, train_data.forget, collator, target).float()
        print(f"    {f_acts.shape[0]} tokens, dim={f_acts.shape[1]}")
        print("  Collecting retain activations...")
        r_acts = collect_activations(model, train_data.retain, collator, target).float()
        print(f"    {r_acts.shape[0]} tokens")
        print("  Collecting wikitext activations...")
        w_acts = collect_activations(model, wikitext_ds, collator, target).float()
        print(f"    {w_acts.shape[0]} tokens")

        # Covariances
        mean_f = f_acts.mean(dim=0)
        Sigma_f = ((f_acts - mean_f).T @ (f_acts - mean_f)) / (f_acts.shape[0] - 1)
        mean_r = r_acts.mean(dim=0)
        Sigma_r = ((r_acts - mean_r).T @ (r_acts - mean_r)) / (r_acts.shape[0] - 1)

        # --- 1. Generalised eigenvalue spectrum ---
        print("  Solving generalised eigenvalue problem...")
        gen_evals, gen_evecs = solve_generalised_eigenvalue(Sigma_f.cuda(), Sigma_r.cuda())
        gen_evals, gen_evecs = gen_evals.cpu(), gen_evecs.cpu()

        print(f"    Top-5 λ:    {[f'{v:.2f}' for v in gen_evals[:5].tolist()]}")
        print(f"    Bottom-5 λ: {[f'{v:.4f}' for v in gen_evals[-5:].tolist()]}")
        print(f"    Range: {gen_evals[0]:.1f} / {gen_evals[-1]:.4f} = {(gen_evals[0]/gen_evals[-1]):.0f}x")
        for t in [2, 5, 10]:
            print(f"    λ > {t}: {(gen_evals > t).sum().item()} directions")

        plot_eigenvalue_spectrum(gen_evals,
            PLOT_DIR / f"disco_eigenspectrum_L{layer_idx}_{task_name}.pdf",
            f"Layer {layer_idx}: Generalised eigenvalue spectrum")

        # --- 2. Tiered variance ratios ---
        print("  Tiered variance ratios:")
        f_tiers = compute_tiered_variance(f_acts, gen_evecs, mean_f, TIERS)
        r_tiers = compute_tiered_variance(r_acts, gen_evecs, mean_f, TIERS)
        w_tiers = compute_tiered_variance(w_acts, gen_evecs, mean_f, TIERS)

        tier_data = {}
        for t in TIERS:
            f, r = f_tiers[t], r_tiers[t]
            ratio = f / r if r > 1e-10 else float("inf")
            tier_data[t] = {"forget": f, "retain": r, "wikitext": w_tiers[t], "ratio": ratio}
            print(f"    {t:>7}: forget={f:.4f}  retain={r:.4f}  ratio={ratio:.2f}x")

        plot_tiered_ratios(tier_data,
            PLOT_DIR / f"disco_tiered_ratios_L{layer_idx}_{task_name}.pdf",
            f"Layer {layer_idx}: Variance fraction per tier")

        # --- 3. PCA vs DISCO ---
        print("  PCA vs DISCO:")
        _, _, V_pca = pt.svd_lowrank(Sigma_f.cuda(), q=min(300, Sigma_f.shape[0]))
        V_pca = V_pca.cpu()
        pca_lambdas = (V_pca.T @ Sigma_f @ V_pca).diag() / (V_pca.T @ Sigma_r @ V_pca).diag().clamp(min=1e-8)

        print(f"    PCA  top-10 mean λ: {pca_lambdas[:10].mean():.2f}")
        print(f"    DISCO top-10 mean λ: {gen_evals[:10].mean():.2f}")
        print(f"    DISCO advantage: {gen_evals[:10].mean() / pca_lambdas[:10].mean():.1f}x")

        plot_pca_vs_disco(pca_lambdas, gen_evals,
            PLOT_DIR / f"disco_vs_pca_L{layer_idx}_{task_name}.pdf",
            f"Layer {layer_idx}: λ of PCA vs DISCO directions")

        # --- 4. Token projection ---
        print("  Token projection:")
        lm_head = model.lm_head.weight.data.float().cpu()
        for label, idxs in [("high_λ", slice(0, 5)), ("low_λ", slice(-5, None))]:
            dirs = gen_evecs[:, idxs]
            logits = lm_head @ dirs
            for i in range(dirs.shape[1]):
                abs_idx = list(range(gen_evecs.shape[1]))[idxs][i]
                lam = gen_evals[abs_idx].item()
                words = [tokenizer.decode(t) for t in logits[:, i].topk(5).indices]
                print(f"    {label} dir {abs_idx} (λ={lam:.2f}): {words}")

        all_results[f"layer_{layer_idx}"] = {
            "eigenvalues_top50": gen_evals[:50].tolist(),
            "eigenvalues_bot50": gen_evals[-50:].tolist(),
            "tiers": tier_data,
            "pca_top10_mean_lambda": pca_lambdas[:10].mean().item(),
            "disco_top10_mean_lambda": gen_evals[:10].mean().item(),
        }

        del f_acts, r_acts, w_acts, Sigma_f, Sigma_r
        pt.cuda.empty_cache()

    save_json(all_results, DISCO_DIR / f"disco_motivation_{task_name}.json")


if __name__ == "__main__":
    main()
