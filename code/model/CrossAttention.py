import torch
import torch.nn as nn
from torch.nn import MultiheadAttention

class CrossAttentionBlock(nn.Module):
    """
    Cross‑attend antigen (queries) over antibody (keys/values).
    Returns an [N_antigen × N_antibody] attention weight matrix.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        # batch_first=True yields (B, L, D) in/out
        self.attn = MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, ag: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
        """
        ag: [N,D], ab: [M,D]
        returns: [N,M] attention weights
        """
        # add batch dim
        ag_b = ag.unsqueeze(0)  # [1, N, D]
        ab_b = ab.unsqueeze(0)  # [1, M, D]

        # Query=ag, Key=Value=ab
        # attn_out: [1,N,D], attn_weights: [1,N,M]
        _, weights = self.attn(ag_b, ab_b, ab_b)

        # squeeze batch
        return weights.squeeze(0)  # [N,M]




# import torch
# import torch.nn as nn
# from torch.nn import Linear,MultiheadAttention

# class CrossAttention(nn.Module):
#     def __init__(self, input_size, num_heads):
#         super(CrossAttention, self).__init__()
#         #MultiHead
#         self.MultiHead_1 = MultiheadAttention(embed_dim=input_size, num_heads=num_heads)


#     def forward(self, input1, input2):
#         input1 = input1.unsqueeze(0).transpose(0, 1)
#         input2 = input2.unsqueeze(0).transpose(0, 1)
#         output_1, attention_weights_1 = self.MultiHead_1(input1, input2, input2)
#         attention_weights_1=attention_weights_1.squeeze(0)
#         output_1 = output_1.transpose(0, 1).squeeze(0)

#         return output_1, attention_weights_1
