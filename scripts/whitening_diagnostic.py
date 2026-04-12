"""Diagnostic: per-dimension forget/retain variance structure for whitening analysis.

Computes per-MLP-module stats to understand where whitening fails vs PCA.
"""
import os
import sys
import torch as pt
import logging
from collections import defaultdict

os.environ["HF_HOME"] = "/VData/kebl6672/.cache/huggingface"
sys.path.insert(0, "src")

from transformers import AutoModelForCausalLM, AutoTokenizer
from data.utils import batched, prep_batch
import hydra
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collect_activations(model, dataset, collator, tokenizer, n_batches=10, batch_size=8):
    """Collect MLP activations per layer/module."""
    acts = defaultdict(list)  # key: (layer, module_name) -> list of (N, D) tensors
    hooks = []

    def make_hook(layer_idx, name):
        def hook_fn(module, args, output):
            acts[(layer_idx, name)].append(args[0].detach().float())
        return hook_fn

    for layer_idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        hooks.append(mlp.gate_proj.register_forward_hook(make_hook(layer_idx, "gate")))
        hooks.append(mlp.up_proj.register_forward_hook(make_hook(layer_idx, "up")))
        hooks.append(mlp.down_proj.register_forward_hook(make_hook(layer_idx, "down")))

    batches = list(batched(dataset, batch_size))[:n_batches]
    for batch in batches:
        batch = collator(batch)
        with pt.no_grad():
            model(**prep_batch(batch, model.device))

    for h in hooks:
        h.remove()

    # Flatten per key: (total_tokens, D)
    result = {}
    for key, tensors in acts.items():
        # Flatten batch and seq dims
        flat = [t.reshape(-1, t.shape[-1]) for t in tensors]
        result[key] = pt.cat(flat, dim=0)
    return result


def analyze(forget_acts, retain_acts):
    """Per-dimension diagnostic across layers."""
    layers = sorted(set(k[0] for k in forget_acts.keys()))
    sample_layers = [layers[0], layers[len(layers)//4], layers[len(layers)//2], layers[3*len(layers)//4], layers[-1]]

    print("\n=== PER-DIMENSION DIAGNOSTIC ===\n")
    print(f"{'Layer':>5} {'Mod':>5} | {'D':>5} | {'σ_f range':>15} | {'σ_r range':>15} | "
          f"{'ratio range':>15} | {'ratio>1.5':>8} {'0.8<r<1.5':>10} {'ratio<0.8':>9} | "
          f"{'σ_f dyn.range':>13}")

    for layer_idx in sample_layers:
        for mod_name in ["gate", "up", "down"]:
            key = (layer_idx, mod_name)
            if key not in forget_acts:
                continue
            f_acts = forget_acts[key]
            r_acts = retain_acts[key]

            sigma_f = f_acts.std(dim=0).clamp(min=1e-6)
            sigma_r = r_acts.std(dim=0).clamp(min=1e-6)
            ratio = sigma_f / sigma_r
            D = sigma_f.shape[0]

            n_selective = (ratio > 1.5).sum().item()
            n_shared = ((ratio >= 0.8) & (ratio <= 1.5)).sum().item()
            n_retain_dom = (ratio < 0.8).sum().item()
            dyn_range = sigma_f.max() / sigma_f.min()

            print(f"L{layer_idx:>4} {mod_name:>5} | {D:>5} | "
                  f"[{sigma_f.min():.1f}, {sigma_f.max():.1f}] | "
                  f"[{sigma_r.min():.1f}, {sigma_r.max():.1f}] | "
                  f"[{ratio.min():.2f}, {ratio.max():.2f}] | "
                  f"{n_selective:>8} {n_shared:>10} {n_retain_dom:>9} | "
                  f"{dyn_range:.0f}x")

    # Summary across all layers
    print("\n=== SUMMARY ===\n")
    all_ratios = []
    all_dyn_ranges = []
    for key in forget_acts:
        sigma_f = forget_acts[key].std(dim=0).clamp(min=1e-6)
        sigma_r = retain_acts[key].std(dim=0).clamp(min=1e-6)
        all_ratios.append(sigma_f / sigma_r)
        all_dyn_ranges.append((sigma_f.max() / sigma_f.min()).item())

    all_ratios = pt.cat(all_ratios)
    print(f"Median per-dim ratio: {all_ratios.median():.3f}")
    print(f"Mean per-dim ratio: {all_ratios.mean():.3f}")
    print(f"Fraction ratio > 1.5 (selective): {(all_ratios > 1.5).float().mean():.3f}")
    print(f"Fraction ratio < 0.8 (retain-dominated): {(all_ratios < 0.8).float().mean():.3f}")
    print(f"Avg σ_f dynamic range: {sum(all_dyn_ranges)/len(all_dyn_ranges):.0f}x")
    print(f"Max σ_f dynamic range: {max(all_dyn_ranges):.0f}x")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Model: {args.model}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=pt.bfloat16).cuda().eval()

    # Load forget and retain datasets
    from data.custom_loaders import wmdp_low_mi, load_hf_and_tokenize
    from omegaconf import OmegaConf

    tok_cfg = OmegaConf.create({"max_length": 128})

    forget_cfg = OmegaConf.create({
        "dataset": "bio", "eval_on_all_questions": False,
        "num_examples_per_question": 3, "tokenizer": tok_cfg
    })
    forget_data = wmdp_low_mi(forget_cfg, tokenizer)
    forget_samples = forget_data.get("forget", list(forget_data.values())[0])

    retain_cfg = OmegaConf.create({
        "dataset_name": "retain", "range": [0, 200], "tokenizer": tok_cfg,
        "hf_args": {"path": "m-a-p/FineFineWeb",
                     "data_files": ["biology/biology_000849.jsonl"]}
    })
    retain_data = load_hf_and_tokenize(retain_cfg, tokenizer)
    retain_samples = retain_data["retain"]

    from transformers import DataCollatorWithPadding
    collator = DataCollatorWithPadding(tokenizer)

    print(f"\nForget samples: {len(forget_samples)}, Retain samples: {len(retain_samples)}")

    forget_acts = collect_activations(model, forget_samples, collator, tokenizer, n_batches=20)
    retain_acts = collect_activations(model, retain_samples, collator, tokenizer, n_batches=20)

    analyze(forget_acts, retain_acts)
