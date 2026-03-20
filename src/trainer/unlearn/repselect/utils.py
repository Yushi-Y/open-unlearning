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
