from typing import NamedTuple
import torch
import torch.nn as nn

class OnesLayer(nn.Module):
    def __init__(self, vector_data):
        super().__init__()
        self.magnitude = nn.Parameter(vector_data)
        
    def forward(self, x):
        return x * self.magnitude.view(1, 1, -1)

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

        self.lora_A = nn.Linear(self.in_features, self.rank, bias=self.bias)
        self.lora_B = nn.Linear(self.rank, self.out_features, bias=self.bias)
        self.frozen_W = nn.Linear(self.in_features, self.out_features, bias=self.bias)

        if self.decompose:
            self.lora_magnitude_layer = OnesLayer(torch.ones(self.out_features))

        # make sure no LoRA weights are marked as "missing" in load_state_dict
        def ignore_missing_keys(m: nn.Module, incompatible_keys: NamedTuple):
            incompatible_keys.missing_keys[:] = []
        self.register_load_state_dict_post_hook(ignore_missing_keys)

    def merge_weight(self):
        with torch.no_grad():
            down_weight = self.lora_A.weight
            up_weight = self.lora_B.weight
            lora_weight = up_weight.mm(down_weight) * self.scaling
            combined_weight = self.frozen_W.weight + lora_weight
            
            if self.decompose:
                # Normalize the combined weight
                norm = combined_weight.norm(p=2, dim=1, keepdim=True)
                normalized_weight = combined_weight / norm
                
                # Apply magnitude scaling
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
        key_name = prefix + "weight"
        if key_name in state_dict:
            w_ref = state_dict[key_name]
            self.frozen_W.load_state_dict({"weight": w_ref}, assign=True)

    def forward(self, x: torch.Tensor):
        if self.decompose:
            lora_output = self.lora_B(self.lora_A(self.dropout(x)))
            frozen_output = self.frozen_W(x)
            combined_output = frozen_output + lora_output
            column_norm = (self.frozen_W.weight + self.lora_B.weight @ self.lora_A.weight).norm(p=2, dim=1).detach()
            normalized_output = combined_output / column_norm.view(1, 1, -1)
            return self.lora_magnitude_layer(normalized_output)
        else:
            lora_output = self.lora_B(self.lora_A(self.dropout(x)))
            return self.frozen_W(x) + lora_output * self.scaling

    def __repr__(self) -> str:
        return "{}Linear(in_features={}, out_features={}, r={}, dropout={}, decompose={})".format(
            "DoRA" if self.decompose else "LoRA",
            self.in_features,
            self.out_features,
            self.rank,
            self.dropout.p,
            self.decompose
        )