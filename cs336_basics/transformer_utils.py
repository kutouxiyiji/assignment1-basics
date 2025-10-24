# The util classes for the transformer
#       - in-house Linear
#       - in-house Embedding

# from signal import SIGALRM
import torch
from einops import rearrange, einsum  # pyright: ignore[reportMissingImports], using uv


class MyLiner(torch.nn.Module):
        def __init_subclass__(cls) -> None:
                return super().__init_subclass__()
        
        def __init__(self, in_features: int, out_features: int, device: torch.device =None, dtype: torch.dtype=None, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.in_features = in_features
                self.out_features = out_features
                self.device = device
                self.dtype = dtype
                
                # Initialize weights with proper shape (in_features, out_features)
                w = torch.empty(in_features, out_features, device=device, dtype=dtype)
                sigma = torch.sqrt(torch.tensor(2/(in_features + out_features)))
                torch.nn.init.trunc_normal_(w, a=-3*sigma, b=3*sigma) # _ indicating self operation, w is modified.
                # Register as nn.Parameter
                self.weights = torch.nn.Parameter(w)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
                return einsum(x, self.weights, '... i, i o -> ... o')

class MyEmbedding(torch.nn.Module):

        def __init_subclass__(cls) -> None:
                return super().__init_subclass__()
        
        def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device =None, dtype: torch.dtype =None, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.num_embeddings = num_embeddings
                self.embedding_dim = embedding_dim

                embeddings = torch.empty(num_embeddings, embedding_dim, dtype = dtype, device = device)
                torch.nn.init.trunc_normal_(embeddings, a = -3, b = 3)
                self.embeddings = torch.nn.Parameter(embeddings)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.embeddings[x]

class MyRMSNorm(torch.nn.Module):

        def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device =None, dtype: torch.dtype =None, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.dtype = dtype
                self.device = device
                self.eps = eps
                gains = torch.ones(d_model, device = device, dtype = dtype)
                self.gains = torch.nn.Parameter(gains)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
                rms = torch.sqrt(torch.mean(x**2, dim = -1, keepdim=True) + self.eps)
                normed_x = x / rms
                return einsum(normed_x, self.gains, "... dmodel, dmodel -> ... dmodel")

class MySwiGLU(torch.nn.Module):

        def __init__(self, d_model: int, d_ff: int = None, device: torch.device = None, dtype: torch.dtype = None, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                if not d_ff:
                        target_d_ff = (8 * d_model) // 3
                        d_ff = ((target_d_ff + 32) // 64) * 64
                        if d_ff == 0:
                                d_ff = 4 * d_model
                # FFN(x) = SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x) ⊙ W3x)
                self.linear1 = MyLiner(d_model, d_ff, device, dtype)
                self.linear3 = MyLiner(d_model, d_ff, device, dtype)
                self.linear2 = MyLiner(d_ff, d_model, device, dtype)
        
        def forward(self, x):
                output1 = self.linear1.forward(x)
                silu = output1 * torch.sigmoid(output1)
                output3 = self.linear3.forward(x)
                element_wise_output = silu * output3
                output = self.linear2.forward(element_wise_output)
                return output