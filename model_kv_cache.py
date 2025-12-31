# A sample implementation by Claude

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
        
        # Flash attention check
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # Causal mask for manual attention
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
        
        # KV Cache storage
        self.k_cache = None  # Will store: (B, n_head, seq_len, head_size)
        self.v_cache = None
        self.use_kv_cache = config.use_kv_cache

    def reset_cache(self):
        """Clear the KV cache - call this at start of each generation"""
        self.k_cache = None
        self.v_cache = None

    def forward(self, x):
        """
        x shape: (B, T, C) where:
        - B = batch size
        - T = sequence length (1 if using cache after first token)
        - C = embedding dimension (n_embd)
        """
        B, T, C = x.size()
        
        # ═══════════════════════════════════════════════════════════
        # STEP 1: Compute Q, K, V for current input
        # ═══════════════════════════════════════════════════════════
        # c_attn projects to 3*C, then we split into Q, K, V
        qkv = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)  # Each: (B, T, C)
        
        # Reshape for multi-head attention
        # (B, T, C) → (B, T, n_head, head_size) → (B, n_head, T, head_size)
        head_size = C // self.n_head
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)

        # ═══════════════════════════════════════════════════════════
        # STEP 2: Handle KV Cache
        # ═══════════════════════════════════════════════════════════
        if self.use_kv_cache:
            if self.k_cache is not None:
                # Append new K, V to cache
                # k_cache: (B, nh, past_len, hs) + k: (B, nh, 1, hs)
                # Result: (B, nh, past_len + 1, hs)
                k = torch.cat([self.k_cache, k], dim=2)
                v = torch.cat([self.v_cache, v], dim=2)
            
            # detach() does TWO things:
            # 1. Creates a tensor with NO grad_fn (no history)
            # 2. Sets requires_grad = False
            # Needed given if using cache with backprop - result in backward through the graph twice
            # no big issue for typical training (not using cache) and inference (not backproping) but will be needed in RLHF & online training
            # Update cache for next iteration
            self.k_cache = k.detach()  # detach to prevent gradient accumulation
            self.v_cache = v.detach()
        
        # After cache handling:
        # - q: (B, nh, T_q, hs) where T_q = 1 (current token only)
        # - k: (B, nh, T_kv, hs) where T_kv = total sequence length so far
        # - v: (B, nh, T_kv, hs)
        
        # ═══════════════════════════════════════════════════════════
        # STEP 3: Compute Attention
        # ═══════════════════════════════════════════════════════════
        T_q = q.size(2)   # Query sequence length
        T_kv = k.size(2)  # Key/Value sequence length (includes cache)
        
        if self.flash:
            # IMPORTANT: When using cache, we can't use is_causal=True
            # because Q has length 1 but K,V have length > 1
            # is_causal=True assumes Q and K have same length
            if self.use_kv_cache and self.k_cache is not None and T_q == 1:
                # Single query attending to all cached keys - no mask needed!
                # The new token can attend to all previous tokens
                y = F.scaled_dot_product_attention(
                    q, k, v, 
                    attn_mask=None, 
                    dropout_p=self.dropout if self.training else 0, 
                    is_causal=False  # <-- Important! Not causal when T_q=1
                ) # TODO: understand
            else:
                # First token or no cache - use causal mask
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0,
                    is_causal=True
                )
        else:
            # Manual attention implementation
            # Q @ K^T: (B, nh, T_q, hs) @ (B, nh, hs, T_kv) = (B, nh, T_q, T_kv)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_size))
            
            # Apply causal mask only when NOT using cache with single query
            if not (self.use_kv_cache and self.k_cache is not None and T_q == 1):
                # Need causal mask: each position can only attend to previous positions
                att = att.masked_fill(self.bias[:, :, :T_q, :T_kv] == 0, float('-inf'))
            # When T_q == 1 and using cache: the single query can attend to ALL keys
            # (which are all previous positions) - this is already causal by construction!
            
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            
            # Weighted sum: (B, nh, T_q, T_kv) @ (B, nh, T_kv, hs) = (B, nh, T_q, hs)
            y = att @ v

        # ═══════════════════════════════════════════════════════════
        # STEP 4: Reshape and project output
        # ═══════════════════════════════════════════════════════════
        # (B, nh, T_q, hs) → (B, T_q, nh, hs) → (B, T_q, C)
        y = y.transpose(1, 2).contiguous().view(B, T_q, C)
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
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # Weight tying
        
    def forward(self, idx, targets=None):
        """
        idx: (B, T) token indices
        
        With KV cache during generation:
        - First call: idx has full prompt, process all tokens
        - Subsequent calls: idx has only the new token
        """
        device = idx.device
        b, t = idx.size()
        
        # ═══════════════════════════════════════════════════════════
        # Handle positional encoding with cache
        # ═══════════════════════════════════════════════════════════
        if self.config.use_kv_cache and self._cache_len() > 0:
            # Cache exists: we're generating, only process the last token
            # Position = current cache length (0-indexed position of new token)
            cache_len = self._cache_len()
            pos = torch.arange(cache_len, cache_len + t, dtype=torch.long, device=device)
            # Only take the last token from input
            idx = idx[:, -1:]  # (B, 1)
        else:
            # No cache: process full sequence
            pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        # Get embeddings
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd) - broadcasts
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Pass through transformer blocks
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        
        # Compute logits
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, -1:, :])  # Only last token for generation
            loss = None
            
        return logits, loss
    
    def _cache_len(self):
        """Get current cache length (sequence length stored in cache)"""
        first_block = self.transformer.h[0]
        if first_block.attn.k_cache is not None:
            return first_block.attn.k_cache.size(2)
        return 0

    def reset_cache(self):
        """Reset cache in ALL attention layers"""
        for block in self.transformer.h:
            block.attn.reset_cache()

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate tokens autoregressively.
        
        With KV cache:
        - First iteration: process full prompt, cache K,V
        - Next iterations: process only new token, update cache
        
        Without KV cache:
        - Every iteration: process full sequence (inefficient!)
        """
        if self.config.use_kv_cache:
            self.reset_cache()
            
        for i in range(max_new_tokens):
            # Crop sequence if needed (can't exceed block_size)
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            
            # Get logits for last position and apply temperature
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx


# ═══════════════════════════════════════════════════════════════════════════════
# TEST IT!
# ═══════════════════════════════════════════════════════════════════════════════

print("="*60)
print("KV CACHE DEMONSTRATION")
print("="*60)

# Create input
idx = torch.tensor([[1, 2, 3]])  # Prompt with 3 tokens
print(f"\nInput prompt: {idx}")
print(f"Input shape: {idx.shape}")

# Test WITH cache
print("\n" + "─"*60)
print("WITH KV CACHE:")
print("─"*60)

config_cache = GPTConfig(use_kv_cache=True)
model_cache = GPT(config_cache)
model_cache.eval()  # .eval() disables dropout (important for reproducibility!)

torch.manual_seed(42)
output_cache = model_cache.generate(idx.clone(), max_new_tokens=5)
print(f"Output: {output_cache}")
print(f"Output shape: {output_cache.shape}")

# Test WITHOUT cache
print("\n" + "─"*60)
print("WITHOUT KV CACHE:")
print("─"*60)

config_no_cache = GPTConfig(use_kv_cache=False)
model_no_cache = GPT(config_no_cache)
# Copy weights to ensure same model
model_no_cache.load_state_dict(model_cache.state_dict())
model_no_cache.eval()

torch.manual_seed(42)
output_no_cache = model_no_cache.generate(idx.clone(), max_new_tokens=5)
print(f"Output: {output_no_cache}")
print(f"Output shape: {output_no_cache.shape}")

# Verify outputs match
print("\n" + "─"*60)
print("VERIFICATION:")
print("─"*60)
print(f"Outputs match: {torch.equal(output_cache, output_no_cache)}")

# Show efficiency difference
print("\n" + "─"*60)
print("EFFICIENCY COMPARISON:")
print("─"*60)
print("""
Without cache (generating 5 new tokens from prompt of 3):
  Step 1: Process [t0, t1, t2]           → 3 tokens
  Step 2: Process [t0, t1, t2, t3]       → 4 tokens  
  Step 3: Process [t0, t1, t2, t3, t4]   → 5 tokens
  ...
  Total attention computations: 3+4+5+6+7 = 25 tokens

With cache (same generation):
  Step 1: Process [t0, t1, t2]           → 3 tokens (build cache)
  Step 2: Process [t3] with cache        → 1 token
  Step 3: Process [t4] with cache        → 1 token
  ...
  Total attention computations: 3+1+1+1+1 = 7 tokens
  
  Speedup: ~3.5x for this small example!
  For longer sequences, the speedup is even more dramatic.
""")