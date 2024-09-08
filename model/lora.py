from typing import NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F

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

        self.lora_A = nn.Parameter(torch.randn(rank, in_features) / rank**0.5)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.frozen_W = nn.Parameter(torch.randn(out_features, in_features) / in_features**0.5, requires_grad=False)
        if bias:
            self.bias_param = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias_param', None)

        if self.decompose:
            self.lora_magnitude_layer = OnesLayer(torch.ones(self.out_features))

        def ignore_missing_keys(m: nn.Module, incompatible_keys: NamedTuple):
            incompatible_keys.missing_keys[:] = []
        self.register_load_state_dict_post_hook(ignore_missing_keys)

    def merge_weight(self):
        with torch.no_grad():
            lora_weight = self.lora_B @ self.lora_A * self.scaling
            combined_weight = self.frozen_W + lora_weight
            if self.decompose:
                column_norm = combined_weight.norm(p=2, dim=0, keepdim=True).detach() + 1e-9
                normalized_weight = combined_weight / column_norm
                final_weight = normalized_weight * self.lora_magnitude_layer.magnitude.view(-1, 1)
            else:
                final_weight = combined_weight
        return final_weight

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
        key_name = prefix + "frozen_W"
        if key_name in state_dict:
            self.frozen_W.data.copy_(state_dict[key_name])

    def forward(self, x: torch.Tensor):
        combined_weight = self.frozen_W + self.lora_B @ self.lora_A * self.scaling
        
        if self.decompose:
            column_norm = combined_weight.norm(p=2, dim=0, keepdim=True).detach() + 1e-9
            normalized_weight = combined_weight / column_norm
            output = F.linear(x, normalized_weight)
            output = self.lora_magnitude_layer(output)
        else:
            output = F.linear(x, combined_weight)
        
        if self.bias_param is not None:
            output += self.bias_param

        return output

    def __repr__(self) -> str:
        return "{}Linear(in_features={}, out_features={}, r={}, dropout={}, decompose={})".format(
            "DoRA" if self.decompose else "LoRA",
            self.in_features,
            self.out_features,
            self.rank,
            self.dropout.p,
            self.decompose
        )