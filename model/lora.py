from typing import NamedTuple

import torch
import torch.nn as nn


class OnesLayer(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(shape))

    def forward(self, x):
        return x * self.weight

class LoRALinear(nn.Module):
    """
    Implementation of:
        - LoRA: https://arxiv.org/abs/2106.09685

    Notes:
        - Freezing is handled at the network level, not the layer level.
        - Scaling factor controls relative importance of LoRA skip
          connection versus original frozen weight. General guidance is
          to keep it to 2.0 and sweep over learning rate when changing
          the rank.
    """

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
        assert not bias
        self.bias = bias
        self.rank = rank
        self.scaling = scaling
        self.decompose = decompose

        self.dropout = nn.Dropout(p=dropout)

        self.lora_A = nn.Linear(
            self.in_features,
            self.rank,
            bias=self.bias,
        )
        self.lora_B = nn.Linear(
            self.rank,
            self.out_features,
            bias=self.bias,
        )
        if self.decompose:
            # print(f"Using lora_magnitude in forward: {self.lora_magnitude}")
            #self.lora_magnitude = nn.Parameter(torch.ones(1, self.out_features))
            self.lora_magnitude = OnesLayer((1, self.out_features))

        # frozen_w is same as base_layer 
        self.frozen_W = nn.Linear(self.in_features, self.out_features, bias=self.bias)

        # make sure no LoRA weights are marked as "missing" in load_state_dict
        def ignore_missing_keys(m: nn.Module, incompatible_keys: NamedTuple):
            # empty missing keys in place
            incompatible_keys.missing_keys[:] = []  # type: ignore

        self.register_load_state_dict_post_hook(ignore_missing_keys)

    # def merge_weight(self):
    #     with torch.no_grad():
    #         down_weight = self.lora_A.weight
    #         up_weight = self.lora_B.weight

    #         weight = up_weight.mm(down_weight) * self.scaling

    #         if self.decompose:
    #             # print(f"Using lora_magnitude in forward: {self.lora_magnitude}")
    #             lora_output_norm_weight = weight/(weight.norm(p=2, dim=1, keepdim=True) + 1e-9)
    #             weight = self.lora_magnitude(lora_output_norm_weight)
                

    #         weight += self.frozen_W.weight
    #     return weight
    
    def merge_weight(self):
        with torch.no_grad():
            down_weight = self.lora_A.weight
            up_weight = self.lora_B.weight

            lora = up_weight.mm(down_weight) 

            if self.decompose:
                column_norm = ((self.frozen_W + lora).norm(p=2, dim=1, keepdim=True)+ 1e-9).detach()
                result = self.frozen_W + lora
                result = result/column_norm
                result = self.lora_magnitude(result)*self.scaling
                
            else: 
                weight = self.frozen_W.weight + lora* self.scaling
        return weight    
    


    ###

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

        # full checkpoint
        if key_name in state_dict:
            w_ref = state_dict[key_name]

            # load frozen weights
            self.frozen_W.load_state_dict({"weight": w_ref}, assign=True)

    # def forward(self, x: torch.Tensor):
    #     # print(f"LoRALinear forward - Input shape: {x.shape}")

    #     lora = self.lora_B(self.lora_A(self.dropout(x)))* self.scaling

    #     if self.decompose:
    #         # print(f"Using lora_magnitude in forward: {self.lora_magnitude}")
    #         lora_output_norm_weight = lora/(lora.norm(p=2, dim=1, keepdim=True) + 1e-9)
    #         lora = self.lora_magnitude(lora_output_norm_weight)
    #     result = self.frozen_W(x) + lora
    #     # print(f"LoRALinear forward - Output shape: {result.shape}")

    #     return result 
    
    # def forward(self, x: torch.Tensor):
    #     # print(f"Input shape: {x.shape}") # [8192, 4096]
    #     lora = self.lora_B(self.lora_A(self.dropout(x))) 
    #     # print(f"LoRA output shape: {lora.shape}") #[8192, 4096]

    #     if self.decompose:
    #         frozen_output = self.frozen_W(x)
    #         # print(f"Frozen output shape: {frozen_output.shape}") #torch.Size([8192, 1024])
    #         result = frozen_output + lora
    #         # print(f"Result shape before norm: {result.shape}") #torch.Size([8192, 1024])
    #         column_norm = (result.norm(p=2, dim=1, keepdim=True) + 1e-9).detach()
    #         # print(f"Column norm shape: {column_norm.shape}") #torch.Size([8192, 1])
    #         result = result / column_norm
    #         result = self.lora_magnitude(result) * self.scaling
    #     else:
    #         result = self.frozen_W(x) + lora * self.scaling

    #     #print(f"Final output shape: {result.shape}")
    #     return result

    def forward(self, x: torch.Tensor):
        print(f"Input shape: {x.shape}")
        
        frozen_output = self.frozen_W(x)
        print(f"Frozen output shape: {frozen_output.shape}")
        
        lora_output = self.lora_B(self.lora_A(self.dropout(x)))
        print(f"LoRA output shape: {lora_output.shape}")

        if self.decompose:
            # Compute W + AB
            combined_weight = self.frozen_W.weight + self.lora_B.weight @ self.lora_A.weight * self.scaling
            print(f"Combined weight shape: {combined_weight.shape}")
            
            # Compute ||W + AB||
            column_norm = combined_weight.norm(p=2, dim=0, keepdim=True).detach() + 1e-9
            print(f"Column norm shape: {column_norm.shape}")
            
            # Normalize: (W + AB) / ||W + AB||
            normalized_weight = combined_weight / column_norm
            print(f"Normalized weight shape: {normalized_weight.shape}")
            
            # Apply magnitude: m * ((W + AB) / ||W + AB||) * x
            result = self.lora_magnitude(normalized_weight.T @ x.T).T
            print(f"Result shape (decompose): {result.shape}")
        else:
            result = frozen_output + lora_output * self.scaling
            print(f"Result shape (non-decompose): {result.shape}")

        return result

    





    def __repr__(self) -> str:
        return "{}Linear(in_features={}, out_features={}, r={}, dropout={}, decompose={})".format(
            "LoRA", self.in_features, self.out_features, self.rank, self.dropout.p, self.decompose
        )
