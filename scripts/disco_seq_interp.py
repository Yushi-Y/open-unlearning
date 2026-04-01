"""
DISCO Sequence Interpretability: What sequences activate each discriminative direction?

Projects forget sequences onto high-λ and low-λ generalised eigenvectors.
High-λ directions should activate on distinctively harmful content;
low-λ directions on generic/shared content.

Also projects directions through lm_head for token-level interpretation.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/disco_seq_interp.py \
        --config-name=unlearn.yaml \
        experiment=unlearn/wmdp_low_mi/default \
        model=Qwen3-8B-Base \
        task_name=DISCO_SEQ_INTERP_QWEN3_8B_BIO
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import hydra
import torch as pt
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from data import get_collators, get_data
from model import get_model
from trainer.utils import seed_everything
from interp_utils import (
    PLOT_DIR, collect_activations, select_layers,
    solve_generalised_eigenvalue, save_json,
)

SAVE_DIR = Path(__file__).resolve().parent.parent / "saves" / "disco_motivation"


def collect_last_hidden(model, dataset, collator, target_module):
    """Collect last-token hidden state per sequence + decoded text."""
    loader = DataLoader(dataset, batch_size=8, collate_fn=collator, shuffle=False)
    captured = []

    def hook(module, args, output):
        captured.append(args[0].detach())

    handle = target_module.register_forward_hook(hook)
    all_acts = []
    all_texts = []

    model.eval()
    with pt.no_grad():
        for batch in loader:
            captured.clear()
            inputs = {
                "input_ids": batch["input_ids"].to(model.device),
                "attention_mask": batch["attention_mask"].to(model.device),
            }
            _ = model(**inputs)
            if not captured:
                continue
            acts = captured[0].float()
            mask = batch["attention_mask"].bool()
            for seq_idx in range(acts.shape[0]):
                last_pos = mask[seq_idx].sum() - 1
                all_acts.append(acts[seq_idx, last_pos].cpu())
                # Decode the sequence
                tokens = batch["input_ids"][seq_idx][mask[seq_idx]]
                all_texts.append(tokens)

    handle.remove()
    return pt.stack(all_acts), all_texts


def truncate_text(text, max_len=120):
    return text[:max_len] + "..." if len(text) > max_len else text


@hydra.main(version_base=None, config_path="../configs", config_name="unlearn.yaml")
def main(cfg: DictConfig):
    seed_everything(cfg.trainer.args.seed)
    task_name = cfg.get("task_name", "DISCO_SEQ_INTERP")

    model, tokenizer = get_model(cfg.model)
    model = model.cuda().eval()

    data = get_data(cfg.data, mode="unlearn", tokenizer=tokenizer, template_args=cfg.model.template_args)
    collator = get_collators(cfg.collator, tokenizer=tokenizer)
    train_data = data["train"]

    layers = select_layers(len(model.model.layers))
    # Focus on middle layer (most discriminative)
    mid_layer = layers[len(layers) // 2]

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    N_DIRS = 5    # directions to show
    N_SEQS = 5    # sequences per direction
    N_TOKENS = 10  # tokens per direction (lm_head projection)

    for layer_idx in [mid_layer]:
        target = model.model.layers[layer_idx].mlp.gate_proj
        print(f"\n{'='*60}")
        print(f"Layer {layer_idx}")
        print(f"{'='*60}")

        # Collect per-sequence activations (last token)
        print("  Collecting forget sequence activations...")
        forget_acts, forget_token_ids = collect_last_hidden(model, train_data.forget, collator, target)
        print(f"    {forget_acts.shape[0]} sequences")

        # Collect all-token activations for covariance
        print("  Collecting all-token activations for covariance...")
        f_acts_all = collect_activations(model, train_data.forget, collator, target).float()
        r_acts_all = collect_activations(model, train_data.retain, collator, target).float()

        mean_f = f_acts_all.mean(dim=0)
        Sigma_f = ((f_acts_all - mean_f).T @ (f_acts_all - mean_f)) / (f_acts_all.shape[0] - 1)
        mean_r = r_acts_all.mean(dim=0)
        Sigma_r = ((r_acts_all - mean_r).T @ (r_acts_all - mean_r)) / (r_acts_all.shape[0] - 1)

        del f_acts_all, r_acts_all
        pt.cuda.empty_cache()

        # Solve generalised eigenvalue problem
        print("  Solving generalised eigenvalue problem...")
        gen_evals, gen_evecs = solve_generalised_eigenvalue(Sigma_f.cuda(), Sigma_r.cuda())
        gen_evals, gen_evecs = gen_evals.cpu(), gen_evecs.cpu()

        # Project forget sequences onto directions
        centered_seqs = forget_acts - mean_f
        projections = centered_seqs @ gen_evecs  # (n_seqs, D)

        # --- Top sequences per direction ---
        print(f"\n  === Top forget sequences per discriminative direction ===")

        high_dirs = list(range(N_DIRS))
        low_dirs = list(range(gen_evecs.shape[1] - N_DIRS, gen_evecs.shape[1]))

        layer_results = {"high_lambda": [], "low_lambda": []}

        for label, dir_indices in [("HIGH-λ", high_dirs), ("LOW-λ", low_dirs)]:
            print(f"\n  --- {label} directions ---")
            for d in dir_indices:
                lam = gen_evals[d].item()
                scores = projections[:, d].abs()
                top_seq_indices = scores.topk(N_SEQS).indices

                print(f"\n    Direction {d} (λ={lam:.2f}):")
                dir_results = {"dir_idx": d, "lambda": lam, "sequences": []}
                for rank, seq_idx in enumerate(top_seq_indices):
                    text = tokenizer.decode(forget_token_ids[seq_idx], skip_special_tokens=True)
                    score = scores[seq_idx].item()
                    print(f"      [{rank+1}] (score={score:.2f}) {truncate_text(text)}")
                    dir_results["sequences"].append({
                        "rank": rank + 1,
                        "score": score,
                        "text": truncate_text(text, 200),
                    })

                key = "high_lambda" if label == "HIGH-λ" else "low_lambda"
                layer_results[key].append(dir_results)

        # --- Token projection through lm_head ---
        print(f"\n  === Token projection through lm_head ===")
        lm_head = model.lm_head.weight.data.float().cpu()

        token_results = {"high_lambda": [], "low_lambda": []}
        for label, dir_indices in [("HIGH-λ", high_dirs), ("LOW-λ", low_dirs)]:
            print(f"\n  --- {label} directions ---")
            for d in dir_indices:
                lam = gen_evals[d].item()
                direction = gen_evecs[:, d]
                logits = lm_head @ direction

                top_tokens = logits.topk(N_TOKENS).indices
                top_words = [tokenizer.decode(t) for t in top_tokens]
                bot_tokens = logits.topk(N_TOKENS, largest=False).indices
                bot_words = [tokenizer.decode(t) for t in bot_tokens]

                print(f"    Dir {d} (λ={lam:.2f})")
                print(f"      Top:    {top_words}")
                print(f"      Bottom: {bot_words}")

                key = "high_lambda" if label == "HIGH-λ" else "low_lambda"
                token_results[key].append({
                    "dir_idx": d, "lambda": lam,
                    "top_tokens": top_words, "bottom_tokens": bot_words,
                })

        all_results[f"layer_{layer_idx}"] = {
            "sequences": layer_results,
            "tokens": token_results,
        }

        del Sigma_f, Sigma_r
        pt.cuda.empty_cache()

    save_json(all_results, SAVE_DIR / f"disco_seq_interp_{task_name}.json")


if __name__ == "__main__":
    main()
