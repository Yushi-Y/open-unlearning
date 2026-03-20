import torch as pt
import torch.nn.functional as F

from data.custom_loaders import DATE_STRING


def get_banned_tokens(processing_class):
    empty_chat = [
        {"role": "system", "content": ""},
        {"role": "user", "content": ""},
        {"role": "assistant", "content": ""},
    ]
    return set(
        processing_class.apply_chat_template(empty_chat, date_string=DATE_STRING)
    )


def npo_saturating_loss(output, batch, beta):
    logits = output.logits
    labels = batch["labels"].to(logits.device)
    shifted_labels = labels[..., 1:].contiguous()
    shifted_logits = logits[..., :-1, :].contiguous()
    per_token_nll = F.cross_entropy(
        shifted_logits.transpose(-1, -2),
        shifted_labels,
        ignore_index=-100,
        reduction="none",
    )
    per_seq_nll = per_token_nll.sum(dim=-1)

    if "initial_nll" not in batch:
        batch["initial_nll"] = per_seq_nll.detach()

    diff = per_seq_nll - batch["initial_nll"].to(per_seq_nll.device)
    return -F.logsigmoid(beta * diff).mean() / beta


class ManualLoRA(pt.nn.Module):
    def __init__(self, in_features, out_features, rank, alpha=16):
        super().__init__()
        self.scaling = alpha / rank
        self.lora_A = pt.nn.Parameter(pt.empty(in_features, rank))
        self.lora_B = pt.nn.Parameter(pt.zeros(rank, out_features))
        pt.nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        self.requires_grad_(True)

    def forward(self, x):
        return x @ self.lora_A @ self.lora_B * self.scaling
