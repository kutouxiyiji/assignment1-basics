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


class RotaryPositionalEmbedding(torch.nn.Module):
        def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.theta = theta
                self.d_k = d_k
                self.max_seq_len = max_seq_len
                self.device = device

                # Theta_i,k = i/Theta^(2*k / d_k), w/ k belongs [0, d_k/2]
                # theta_i,k should be in shape of (max_seq_len, d_k/2)
                if d_k % 2 != 0:
                        raise ValueError("d_k must be even number.")
                # theta_k is shape (d_k//2, )
                theta_k = self.theta ** (-(torch.arange(0, self.d_k, 2, dtype = torch.float32, device = self.device)/self.d_k))
                # seq_pos is shape (max_seq_len, )
                seq_pos = torch.arange(max_seq_len, dtype=torch.float32, device = self.device)
                theta_ik = einsum(seq_pos, theta_k, "n,d->n d")
                # cos, sin, in shape (max_seq_len, d_k/2)
                cos_theta = torch.cos(theta_ik)
                sin_theta = torch.sin(theta_ik)
                # cast to (max_seq_len, d_k)
                cos_vals = torch.repeat_interleave(cos_theta, 2, dim = -1)
                sin_vals = torch.repeat_interleave(sin_theta, 2, dim = -1)
                self.register_buffer("cos_cache", cos_vals, persistent=False)
                self.register_buffer("sin_cache", sin_vals, persistent=False)
        
        def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
                # equations
                # for pos i, the dim 2k and 2k+1
                # [x_2k, x_2k + 1]_rotated = [row#1: cos(theta_k), -sin(theta_k); row#2: sin(theta_k), cos(theta_k)] * [x_2k, x_2k + 1]
                # which equals to x_2k_rotated = x_2k* cos - x_2k+1 * sin
                # x_2k+1_rotated = x_2k*sin + x_2k+1 * cos

                # get the cos and sin from buffer
                # cos and sin will be (..., seq_len, d_k) shape
                cos = self.cos_cache[token_positions]
                sin = self.sin_cache[token_positions]
                # crate a x_rot which will cast x_2k to -x_2k+1 (for even idx) and x_2k+1 to x_2k (for odd idx)
                # then the equaiton above will become
                # x * cos + x_rot * sin
                x_rot = torch.empty_like(x)
                # even idx
                x_rot[..., 0::2] = -x[..., 1::2]
                x_rot[..., 1::2] = x[..., 0::2]
                # element wise multiply, both shape is (..., seq, d_k).
                return x*cos + x_rot*sin

def mySoftMax(x: torch.Tensor, i: int) -> torch.Tensor:
        max_x, _ = torch.max(x, dim= i, keepdim=True)
        shifted_x_exp = torch.exp(x - max_x)
        sum_exp = torch.sum(shifted_x_exp, dim= i, keepdim=True)
        return shifted_x_exp / sum_exp
        

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None):
        # Q is shape (n, d), K is shape (m, d) and V is shape (m, d).
        d_k = Q.shape[-1]
        QK = einsum(Q, K, "... Query_len dmodel, ... Key_len dmodel -> ... Query_len Key_len")
        QK_norm = QK / torch.sqrt(torch.tensor(d_k, dtype=Q.dtype, device=Q.device))
        if mask is not None:
                QK_norm = QK_norm + torch.where(mask, 0, float("-inf"))
        return mySoftMax(QK_norm, -1) @ V


class MyMultiHeadAttention(torch.nn.Module):

        def __init__(self, d_model:int, num_heads:int, d_in:int = None, d_out:int = None, theta:float = None, max_seq_len: int = None, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.d_model = d_model
                self.num_heads = num_heads
                self.max_seq_len = max_seq_len
                if d_model % num_heads != 0:
                        raise ValueError("d_k should be an interger!")
                self.d_k = d_model // num_heads
                #input (... d_in), usually d_in is d_k
                # Q heads, ... d_in, d_in d_k -> ... d_k
                if d_in is None:
                        d_in = self.d_model
                self.d_in = d_in
                self.q_heads = torch.nn.ModuleList([MyLiner(self.d_in, self.d_k) for _ in range(self.num_heads)])
                self.k_heads = torch.nn.ModuleList([MyLiner(self.d_in, self.d_k) for _ in range(self.num_heads)])
                self.v_heads = torch.nn.ModuleList([MyLiner(self.d_in, self.d_k) for _ in range(self.num_heads)])
                if d_out is None:
                        self.d_out = self.d_model
                self.attention_output = MyLiner(self.d_model, self.d_out)
                self.rope = None
                if theta is not None and max_seq_len is not None:
                        self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len)
        
        def forward(self, x: torch.Tensor):
                seq_len = x.shape[-2]
                # Create causal mask: lower triangular (seq_len, seq_len)
                causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
                
                # Expand mask to match batch dimensions: (..., seq_len, seq_len)
                batch_dims = x.shape[:-2]  # All dimensions before (seq_len, d_in)
                for _ in batch_dims:
                        # Adds a leading dimension to causal_mask once per batch dimension.
                        causal_mask = causal_mask.unsqueeze(0)
                # Final shape: (batch_size, ..., seq_len, seq_len).
                causal_mask = causal_mask.expand(*batch_dims, seq_len, seq_len)
                
                multi_head_attentions = []
                # Create token positions matching batch dimensions: (..., seq_len)
                # For batched input (batch, seq_len, d_in), positions should be (batch, seq_len)
                token_positions_base = torch.arange(seq_len, device=x.device)
                # Expand to match batch dimensions if needed
                if len(batch_dims) > 0:
                        token_positions = token_positions_base.unsqueeze(0).expand(*batch_dims, seq_len)
                else:
                        token_positions = token_positions_base
                
                for head in range(self.num_heads):
                        Q = self.q_heads[head].forward(x) # W*In = Q_i
                        K = self.k_heads[head].forward(x)
                        if self.rope is not None:
                                Q = self.rope.forward(Q, token_positions)
                                K = self.rope.forward(K, token_positions)
                        V = self.v_heads[head].forward(x)
                        multi_head_attentions.append(scaled_dot_product_attention(Q, K, V, causal_mask))
                concat_attentions = torch.concat(multi_head_attentions, dim = -1)
                return self.attention_output.forward(concat_attentions)

class MyTransformerBlock(torch.nn.Module):

        def __init__(self, d_model:int, num_heads:int, d_ff:int, theta:float = None, max_seq_len: int = None, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.num_heads = num_heads
                self.d_ff = d_ff
                self.d_model = d_model
                self.rmsn1 = MyRMSNorm(d_model)
                self.multi_attentions = MyMultiHeadAttention(d_model, num_heads, theta=theta, max_seq_len=max_seq_len)
                self.rmsn2 = MyRMSNorm(d_model)
                self.swiglu = MySwiGLU(d_model, d_ff)
        
        def forward(self, x: torch.Tensor):
                norm_x = self.rmsn1.forward(x)
                attentions = self.multi_attentions(norm_x)
                attention_sublayer_output = x + attentions
                norm_attention_output = self.rmsn2.forward(attention_sublayer_output)
                ffn_output = self.swiglu.forward(norm_attention_output)
                return attention_sublayer_output + ffn_output