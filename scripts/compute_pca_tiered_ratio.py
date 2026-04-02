"""Compute forget/retain variance ratio along PCA directions, tiered."""
import torch as pt
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings, os, sys
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-3B', dtype=pt.bfloat16, device_map='cuda')
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B')
tokenizer.pad_token = tokenizer.eos_token

forget_ds = load_from_disk(".cache/load_hf/filypo_wmdp_bio_T_train_None")
retain_ds = load_from_disk(".cache/load_hf/m-a-p_FineFineWeb_train_['biology_biology_000849.jsonl']")

results = []
for li in [0, 9, 18, 27]:
    mod = model.model.layers[li].mlp.gate_proj
    fa, ra = [], []

    def hk(m, i, o):
        hk.l = i[0].detach().reshape(-1, i[0].shape[-1])
    h = mod.register_forward_hook(hk)

    for i in range(50):
        b = {k: pt.tensor(v).unsqueeze(0).cuda() for k, v in forget_ds[i].items() if k in ['input_ids', 'attention_mask']}
        with pt.no_grad(): model(**b)
        fa.append(hk.l.cpu())

    for i in range(50):
        b = {k: pt.tensor(v).unsqueeze(0).cuda() for k, v in retain_ds[i].items() if k in ['input_ids', 'attention_mask']}
        with pt.no_grad(): model(**b)
        ra.append(hk.l.cpu())

    h.remove()

    F = pt.cat(fa).float()
    R = pt.cat(ra).float()
    Sf = pt.cov(F.T)
    Sr = pt.cov(R.T)
    _, S, V = pt.svd_lowrank(Sf, q=400)
    fv = (V.T @ Sf @ V).diagonal()
    rv = (V.T @ Sr @ V).diagonal().clamp(min=1e-4)
    ratio = fv / rv

    for lo, hi in [(0, 3), (3, 10), (10, 30), (30, 100), (100, 300), (300, 400)]:
        r = ratio[lo:hi].mean().item()
        results.append((li, lo, hi, r))

print("\nTiered forget/retain ratio along PCA directions (Llama-3.2-3B, WMDP-Bio, gate_proj)")
print(f"{'Tier':<12} {'L0':>6} {'L9':>6} {'L18':>6} {'L27':>6}")
tiers = [(0,3), (3,10), (10,30), (30,100), (100,300), (300,400)]
for lo, hi in tiers:
    vals = [r for li, l, h, r in results if l == lo and h == hi]
    print(f"{lo}-{hi:<8} {vals[0]:>5.1f}x {vals[1]:>5.1f}x {vals[2]:>5.1f}x {vals[3]:>5.1f}x")
