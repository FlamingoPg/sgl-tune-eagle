        import torch.nn.functional as F
        
        # Repeat k and v to match the number of query heads
        k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        
        out = F.scaled_dot_product_attention(
            q.view(-1, self.num_heads, self.head_dim).transpose(1, 2),
            k.view(-1, self.num_heads, self.head_dim).transpose(1, 2),
            v.view(-1, self.num_heads, self.head_dim).transpose(1, 2),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )