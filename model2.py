import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int =32000 #  -1 #set when we load the tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-06

    #needed for kv cache
    max_batch_size: int = 1
    max_seq_len: int = 1024

    device: str = None
def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device:str, theta: float = 10000.0):
    assert head_dim %2 == 0, 'head_dim must be divisible by 2.'
    theta_numerator = torch.arange(0, head_dim, 2, dtype=torch.float) #(head_size/2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    #construct the m parameters
    m = torch.arange(seq_len, device=device)
    # Shape: (Seq_len) outer_p (head_dim//2) --> (seq_len, head_dim//2)
    frequency = torch.outer(m, theta).float()
    # (Seq_len) outer_p (head_dim//2) --> (Seq_len, head_dim//2)
    freqs_complex = torch.polar(torch.ones_like(frequency), frequency)
    return  freqs_complex
def apply_rotary_embeddings(x: torch.Tensor, freq_complex: torch.Tensor, device: str):
    # B, Seq_len, H, head_dim ---> (B, seq_len, H, head_dim//2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (Seq_len, head_dim//2)  ---> (1, seq_len, 1, head_dim/2)
    freq_complex = freq_complex.unsqueeze(0).unsqueeze(2)
    #  (B, seq_len, H, head_dim//2) * (1, seq_len, 1, head_dim/2) ---> (B, seq_len, H, head_dim//2, )
    x_rotated = x_complex * freq_complex
    # (B, seq_len, H, head_dim//2) ----> (B, seq_len, H, head_dim//2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, seq_len, H, head_dim//2, 2) ---> (B, seq_len, H, head_dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)
def repeat_kv(x: torch.tensor, n_reps: int) -> torch.tensor:
    # (B, seq_len of kv, n_kv_head, head_dim)
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_reps == 1:
        return x
    return (
        x[:, :, :, None, : ]
        .expand(batch_size,  seq_len, n_kv_heads, n_reps, head_dim )
        .reshape(batch_size, seq_len, n_kv_heads*n_reps, head_dim)
    )

class selfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # number of kv heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # number of heads for queries
        self.n_heads_q = args.n_heads

        self.n_rep = self.n_heads_q//self.n_kv_heads
        # the dimension of each heads
        self.head_dim = args.dim//self.n_heads_q
        self.wq = nn.Linear(args.dim, args.n_heads*self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads*self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads*self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads*self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

        self.cache_idx = 1
        self.max_seq_len = args.max_seq_len

    def  forward(self, x: torch.tensor, start_pos: int, freq_complex: torch.tensor):
        batch_size, seq_len, _ = x.shape # (B, 1, dim)
        # (B, 1, Dim) -> (B, 1, n_heads*head_dim)
        xq = self.wq(x)
        # (B, 1, Dim) -> (B, 1, n_kv_head*head_dim)
        xk = self.wk(x)
        # (B, 1, Dim) -> (B, 1, n_kv_head*head_dim)
        xv = self.wv(x)
        # (B, 1, n_heads*head_dim) -> (B, 1, n_heads, head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (B, 1, n_kv_head*head_dim) -> (B, 1, n_kv_head, head_dim )
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (B, 1, n_kv_head*head_dim) -> (B, 1, n_kv_head, head_dim )
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # (B, 1, n_heeads, 1, head_dim) -> (B, 1, n_heeads,  head_dim)
        xq = apply_rotary_embeddings(xq, freq_complex[self.cache_idx:self.cache_idx+seq_len,:], device=x.device)
        # (B, 1, n_kv_heads, head_dim) -> (B, 1, n_kv_heads,  head_dim)
        #xk = apply_rotary_embeddings(xk, freq_complex[self.cache_idx:self.cache_idx+seq_len], device=x.device)
        # (B, seq_len of kv, n_kv_head, head_dim)
        self.cache_k[:batch_size, self.cache_idx:self.cache_idx+seq_len] = xk
        self.cache_v[:batch_size, self.cache_idx:self.cache_idx+seq_len] = xv

        #rotate all the cached key vectors
        keys = apply_rotary_embeddings(
            self.cache_k[:batch_size, :self.cache_idx+seq_len], freq_complex[:self.cache_idx+seq_len, :], device = x.device
                                                                             )

        #keys = self.cache_k[:batch_size, :self.cache_idx+seq_len]
        values = self.cache_v[:batch_size, :self.cache_idx+seq_len]
        self.cache_idx += 1
        # duplicate the heads of k and v n_rep(total_head//kv_head times) to use for each q
        # # (B, seq_len of kv, n_kv_head, head_dim) ->batch_size, seq_len, n_kv_heads*n_reps, head_dim
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # (B, 1, n_heads, head_dim) --> (B, n_heds, 1, head_dim)
        xq = xq.transpose(1, 2)
        #(B, seq_len_kv, n_heads, head_dim) --> (B, n_heds, seq_len_kv, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        #(B, n_heds, 1, head_dim) @ (B, n_heds, head_dim, seq_len_kv) -> (B, n_heads, 1, seq_len_kv)
        scores = torch.matmul(xq, keys.transpose(2, 3))/math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # (B, n_heads, 1, seq_len_kv)@(B, n_heds, seq_len_kv, head_dim) -> (B, n_heds, 1, head_dim)
        out = torch.matmul(scores, values)
        # (B, n_heds, 1, head_dim) --> (B, 1, n_heads, head_dim) --> (B, seq_len, Dim)
        out = (out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))

        return self.wo(out)
    def clean_kv_cache(self,  space_needed=500, recent_size=1020):
        if self.cache_idx + space_needed <= self.max_seq_len:
            return
        start_idx = self.cache_idx - recent_size + space_needed
        end_idx = self.cache_idx
        temp = torch.cat([
            self.cache_k[:, :4],
            self.cache_k[:, start_idx:end_idx]
        ], dim=1)
        self.cache_k[:, :temp.shape[1]] = temp 
        temp= torch.cat([
            self.cache_v[:, :4],
            self.cache_v[:, start_idx:end_idx]
        ], dim=1)
        self.cache_v[:, :temp.shape[1]] = temp
        self.cache_idx = 4 + end_idx - start_idx

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = args.dim*4
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)
    def forward(self, x: torch.tensor):
        swish  = F.silu(self.w1(x))
        x_v = self.w3(x)
        x = self.w2(swish * x_v)
        return x


class  EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = self.dim//self.n_heads
        self.attention = selfAttention(args)
        self.feed_forward = FeedForward(args)
        # RMSNorm before attention block
        self.attention_norm = RMSNorm(self.dim, args.norm_eps)
        # RMSNorm before the ffw layer)
        self.ffn_norm = RMSNorm(self.dim, args.norm_eps)
    def forward(self, x: torch.tensor, start_pos: int, freq_complex: torch.tensor):
        # (B, seq_len, dim) + (B, seq_len, dim) ---> (B, seq_len, dim)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freq_complex)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
class RMSNorm(nn.Module):
    def __init__(self, dim: int,  eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self, x: torch.tensor):
        # (B, seq_len, dim) * (B, seq_len, 1) ----> (B, Seq_len, dim)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x: torch.tensor):
        # (Dim) * (B, seq_len, head_dim) ---> (B, seq_len, head_dim)
        return self.weight*self._norm(x.float()).type_as(x)
class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size != - 1, 'Vocab size must be set'
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size,  args.dim)
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(EncoderBlock(args))
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim//self.args.n_heads, self.args.max_seq_len*2, device=self.args.device )

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (B, seq_len)
        B, seq_len = tokens.shape
        assert seq_len == 1, 'only one token at a time can be processed'
        #(B, seq_len) --> (B, seq_len, dim)
        h = self.tok_embeddings(tokens)

        # retreive the pairs(m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex

        #consecutively apply to all encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)

        h = self.norm(h)
        output = self.output(h).float()
        return output
