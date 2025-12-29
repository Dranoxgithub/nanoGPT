# An implementation by Anni

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
        self.cache = None
        self.use_kv_cache = config.use_kv_cache

    def reset_cache(self):
        self.cache = None
    def forward(self, x):
        B, T, C = x.size()
        
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2) # (B, 1, C)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, 1, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.use_kv_cache and self.cache:
            k_cache, v_cache = self.cache[0], self.cache[1] # (B, nh, T - 1, C)
            k = torch.concat((k_cache, k), dim= 2)
            v = torch.concat((v_cache, v), dim= 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # q of shape (B, nh, 1, hs); k of shape (B, nh, T, hs); (B, nh, 1, T)
            if self.use_kv_cache and self.cache:
                att = att
            else:
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1) 
            att = self.attn_dropout(att) # (B, nh, 1, T)
            y = att @ v # (B, nh, 1, C)

        if self.use_kv_cache:
            self.cache = [k, v]

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 64
    vocab_size: int = 100
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True
    use_kv_cache: bool = False

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for i in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        if self.config.use_kv_cache and self.cache_exists():
            pos = torch.tensor([t - 1], dtype=torch.long, device=device)
            idx = idx[:, [-1]]
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :]) # (B, 1, vocab_size)
            loss = None
        return logits, loss
    def cache_exists(self):
        for block in self.transformer.h:
            if block.attn.cache != None: 
                return True
        return False

    def reset_cache(self):
        """Reset cache in ALL attention layers"""
        for block in self.transformer.h:
            block.attn.reset_cache()

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        if self.config.use_kv_cache:
            self.reset_cache()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf') # TODO: where's top_p
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ============== Test It! ==============
config = GPTConfig(use_kv_cache=True)
model = GPT(config)
model.eval() # TODO: why .eval since there's @torch.no_grad() alr

print("Model created!")
print(f"Config: {config}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test generate
torch.manual_seed(42)
output = model.generate(idx, max_new_tokens=2)
print(f"\nGenerated shape: {output.shape}")
print(f"Generated tokens: {output}")

print("-------------")
torch.manual_seed(42)
config = GPTConfig(use_kv_cache=False)
model_without_cache = GPT(config)
model.eval()

torch.manual_seed(42)
output = model_without_cache.generate(idx, max_new_tokens=2)
print(f"Generated tokens without cache: {output}")