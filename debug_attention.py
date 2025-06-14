import torch
import torch.nn.functional as F

def debug_attention():
    # 设置参数
    batch_size = 2
    seq_len = 4
    embed_dim = 64
    num_heads = 4
    head_dim = embed_dim // num_heads

    # 创建随机输入
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # 创建投影矩阵
    q_proj = torch.nn.Linear(embed_dim, embed_dim)
    k_proj = torch.nn.Linear(embed_dim, embed_dim)
    v_proj = torch.nn.Linear(embed_dim, embed_dim)
    
    # 计算 Q, K, V
    q = q_proj(x)  # [batch_size, seq_len, embed_dim]
    k = k_proj(x)  # [batch_size, seq_len, embed_dim]
    v = v_proj(x)  # [batch_size, seq_len, embed_dim]
    
    # 重塑张量以进行多头注意力计算
    q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
    k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
    v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
    
    # 计算注意力分数
    scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)  # [batch_size, num_heads, seq_len, seq_len]
    
    # 应用 softmax
    attn_weights = F.softmax(scores, dim=-1)
    
    # 计算输出
    output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
    
    # 重塑回原始形状
    output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
    
    # 打印调试信息
    print("\n输入张量形状:")
    print(f"x shape: {x.shape}")
    
    print("\nQ, K, V 形状:")
    print(f"q shape: {q.shape}")
    print(f"k shape: {k.shape}")
    print(f"v shape: {v.shape}")
    
    print("\n注意力分数形状:")
    print(f"scores shape: {scores.shape}")
    
    print("\n注意力权重形状:")
    print(f"attn_weights shape: {attn_weights.shape}")
    
    print("\n输出形状:")
    print(f"output shape: {output.shape}")
    
    # 打印一些具体的值
    print("\nQ 矩阵的第一个头的值:")
    print(q[0, 0])  # 第一个batch，第一个头的Q值
    
    print("\nK 矩阵的第一个头的值:")
    print(k[0, 0])  # 第一个batch，第一个头的K值
    
    print("\n注意力权重的第一个头的值:")
    print(attn_weights[0, 0])  # 第一个batch，第一个头的注意力权重

if __name__ == "__main__":
    debug_attention() 