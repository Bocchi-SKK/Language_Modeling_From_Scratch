import torch.nn as nn
import torch.nn.init as init
import torch
from torch import Tensor
from einops import rearrange,einsum

# Check if CUDA is available and set the default device for torch
if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super(Linear, self).__init__()
        self.in_features:int = in_features # final dimension of the input
        self.out_features:int = out_features # final dimension of the output
        self.device:torch.device = device # Device to store the parameters on
        self.dtype:torch.dtype = dtype # Data type of the parameters

        # Create weight as learnable parameters
        self.weight:nn.Parameter = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )

        # Initialize weight with truncated normal
        std = (2 / (in_features + out_features)) ** 0.5
        init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)

    def forward(self, x:torch.Tensor) -> torch.Tensor: # Apply the linear transformation to the input
        return x @ self.weight.T

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super(Embedding, self).__init__()
        self.num_embeddings:int = num_embeddings # Size of the vocabulary
        self.embedding_dim:int = embedding_dim # Dimension of the embedding vectors
        self.device:torch.device = device # Device to store the parameters on
        self.dtype:torch.dtype = dtype # Data type of the parameters

        # Create the embedding matrix as learnable parameters
        self.weight:nn.Parameter = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )

        # Initialize the embedding matrix with truncated normal
        std = 1
        init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3, b=3)

    def forward(self, token_ids:torch.Tensor) -> torch.Tensor: # Lookup the embedding vectors for the given token IDs
        return self.weight[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model:int, eps:float=1e-5, device=None, dtype=None):
        super(RMSNorm, self).__init__()
        self.d_model:int = d_model # Hidden dimension of the model
        self.eps:float = eps # Epsilon value for numerical stability
        self.device:torch.device = device # Device to store the paramters on
        self.dtype:torch.dtype = dtype # Data type of the parameters

        # Create the learnable scale parameter
        self.weight:nn.Parameter = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor: # Process an input tensor of shape(batch_size, sequence_length, d_model) and return a tensor of the same shape
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # Compute the root mean square of the input tensor
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)

        # Normalize the input tensor
        result = (x / rms) * self.weight
        return result.to(in_dtype)

def SiLU(x:torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super(SwiGLU, self).__init__()
        # Initialize the weights for the SwiGLU module
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(SiLU(self.w1(x)) * self.w3(x))
    
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta:float, d_k:int, max_seq_len:int, device=None):
        super(RotaryPositionalEmbedding, self).__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
        positions = torch.arange(max_seq_len, dtype=torch.float, device=device)
        angles = torch.einsum('i,j->ij', positions, inv_freq)
        self.register_buffer('cos', torch.cos(angles), persistent=False)
        self.register_buffer('sin', torch.sin(angles), persistent=False)

    def forward(self, x:torch.Tensor, token_positions:torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_k)
        # token_positions: (batch, seq_len)
        x1 = x[..., ::2] # All even-indexed features(first of each pair)
        x2 = x[..., 1::2] # All odd-indexed features(second of each pair)
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        x_rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rotated.flatten(-2)
    
def softmax(x:torch.Tensor, dim:int)->torch.Tensor:
    x = x - x.max(dim=dim, keepdim=True).values  # subtract max for stability
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None) -> Tensor:
    d_k = Q.shape[-1]  # Dimension of the keys
    scores = (Q @ K.transpose(-2,-1)) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(~mask, -1e9)  # or -float('inf')
    softmax_scores = softmax(scores, dim=-1)
    attention_output = softmax_scores @ V  # shape: (batch, seq_len, d_v)
    return attention_output

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int, device=None, dtype=None):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        d_k:int = d_model // num_heads  # Dimension of each head (and set d_k = d_v = d_model/num_heads)
        self.d_k = d_k  # Store d_k for later use
        h = num_heads

        self.w_q = Linear(d_model, d_model, device=device, dtype=dtype) # w_q for query projection
        self.w_k = Linear(d_model, d_model, device=device, dtype=dtype) # w_k for key projection
        self.w_v = Linear(d_model, d_model, device=device, dtype=dtype) # w_v for value projection
        self.w_o = Linear(d_model, d_model, device=device, dtype=dtype) # w_o for output projection

    def forward(self, x:Tensor, rope: RotaryPositionalEmbedding = None)->torch.Tensor:
        # x: (batch, seq_len, d_model)
        Q = self.w_q(x)  # (batch, seq_len, num_heads*d_k)
        K = self.w_k(x)
        V = self.w_v(x)

        # Split heads
        Q = rearrange(Q, 'b s (h d) -> b h s d', h=self.num_heads, d=self.d_k)
        K = rearrange(K, 'b s (h d) -> b h s d', h=self.num_heads, d=self.d_k)
        V = rearrange(V, 'b s (h d) -> b h s d', h=self.num_heads, d=self.d_k)

        # Apply RoPE to Q and K if provided
        if rope is not None:
            batch_size, seq_len = x.shape[0], x.shape[1]
            token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
            
            # Apply RoPE to each head
            Q_rope = torch.zeros_like(Q)
            K_rope = torch.zeros_like(K)
            for h in range(self.num_heads):
                Q_rope[:, h] = rope(Q[:, h], token_positions)
                K_rope[:, h] = rope(K[:, h], token_positions)
            Q, K = Q_rope, K_rope

        # Create a mask for the attention scores
        seq_len = x.shape[1]
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)  # Upper triangle, excluding diagonal
        mask = ~mask  # Invert: True where allowed (j <= i), False where masked (j > i)
        # Compute the attention output
        attention_output = scaled_dot_product_attention(Q, K, V, mask=mask)  # (batch, num_heads, seq_len, d_k)
        # Merge heads
        attention_output = rearrange(attention_output, 'b h s d -> b s (h d)')
        output = self.w_o(attention_output)  # (batch, seq_len, d_model)
        return output

class TransformerBlock(nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int, max_seq_len:int, theta:float=None):
        super(TransformerBlock, self).__init__()
        self.normalization1 = RMSNorm(d_model)
        self.normalization2 = RMSNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, num_heads)
        self.ff = SwiGLU(d_model, d_ff) # ff for feed-forward network

        # Create the rotary positional embedding
        d_k:int = d_model // num_heads  # Dimension of each head (and set d_k = d_v = d_model/num_heads)
        if (theta):
            self.rope = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len)
        else:
            self.rope = None

    def forward(self, x:Tensor) -> Tensor:
        # the firset 'sub-layer'
        y = x + self.attention(self.normalization1(x), rope=self.rope)

        # the second 'sub-layer'
        output = y + self.ff(self.normalization2(y))
        return output   

class TransformerLanguageModel(nn.Module):
    def __init__(self,
                 vocab_size:int,
                 context_length:int,
                 d_model:int,
                 num_layers:int,
                 num_heads:int,
                 d_ff:int,
                 rope_theta:float):
        super(TransformerLanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.token_embedding = Embedding(vocab_size, d_model)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_len=context_length, theta=rope_theta) for _ in range(num_layers)])
        self.normalization = RMSNorm(d_model)
        self.output_embedding = Linear(d_model, vocab_size)

    def forward(self, x:Tensor) -> Tensor:
        # x: (batch, seq_len)
        x = self.token_embedding(x)
        # x: (batch, seq_len, d_model)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.normalization(x)
        # x: (batch, seq_len, d_model)
        x = self.output_embedding(x)
        # x: (batch, seq_len, vocab_size)
        return x  # Return the output tensor of shape (batch, seq_len, vocab_size)