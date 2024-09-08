from typing import NamedTuple
import torch
import torch.nn as nn

class OnesLayer(nn.Module):
    def __init__(self, vector_data):
        super().__init__()
        self.magnitude = nn.Parameter(vector_data)
        
    def forward(self, x):
        return x * self.magnitude.view(1, -1)

class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        scaling: float,
        dropout: float,
        bias: bool = False,
        decompose: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.rank = rank
        self.scaling = scaling
        self.dropout = nn.Dropout(p=dropout)
        self.decompose = decompose

        self.lora_A = nn.Linear(self.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, self.out_features, bias=False)
        self.frozen_W = nn.Linear(self.in_features, self.out_features, bias=self.bias)

        if self.decompose:
            self.lora_magnitude_layer = OnesLayer(torch.ones(self.out_features))

        def ignore_missing_keys(m: nn.Module, incompatible_keys: NamedTuple):
            incompatible_keys.missing_keys[:] = []
        self.register_load_state_dict_post_hook(ignore_missing_keys)

    def merge_weight(self):
        with torch.no_grad():
            down_weight = self.lora_A.weight
            up_weight = self.lora_B.weight
            weight = up_weight.mm(down_weight) * self.scaling
            weight += self.frozen_W.weight
        return weight

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        key_name = prefix + "weight"
        if key_name in state_dict:
            w_ref = state_dict[key_name]
            self.frozen_W.load_state_dict({"weight": w_ref}, assign=True)

    def forward(self, x: torch.Tensor):
        frozen_output = self.frozen_W(x)
        lora_output = self.lora_B(self.lora_A(self.dropout(x)))

        if self.decompose:
            combined_weight = self.frozen_W.weight + self.lora_B.weight @ self.lora_A.weight * self.scaling
            column_norm = combined_weight.norm(p=2, dim=0, keepdim=True).detach() + 1e-9
            normalized_weight = combined_weight / column_norm
            combined_output = x @ normalized_weight.T
            return self.lora_magnitude_layer(combined_output)
        else:
            return frozen_output + lora_output * self.scaling

    def __repr__(self) -> str:
        return "{}Linear(in_features={}, out_features={}, r={}, dropout={}, decompose={})".format(
            "DoRA" if self.decompose else "LoRA",
            self.in_features,
            self.out_features,
            self.rank,
            self.dropout.p,
            self.decompose
        )