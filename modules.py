import torch
from torch import nn
import torch.nn.functional as F

def create_lookahead_mask(x: torch.Tensor):
    # x: (B, N, d_model)
    assert len(x.shape) == 3
    b, n, _ = x.shape
    mask = torch.tril(torch.ones(n, n))
    mask = mask.unsqueeze(0).repeat(b, 1, 1).to(x)
    return mask

def binary_mask_to_attention_mask(key_binary_mask: torch.Tensor, query_sequence_length: int):
    # binary_mask: (B, N_kv)
    # mask: (B, N_q, N_kv)
    mask = key_binary_mask.float().unsqueeze(1).repeat(1, query_sequence_length, 1).to(key_binary_mask)
    return mask

class Attention(nn.Module):
    def __init__(self, *, d_model, d_k):
        super().__init__()
        assert d_model % d_k == 0
        self.d_k = d_k # = d_model / n_heads
        self.q_linear = nn.Linear(in_features=d_model, out_features=d_k)
        self.k_linear = nn.Linear(in_features=d_model, out_features=d_k)
        self.v_linear = nn.Linear(in_features=d_model, out_features=d_k)
    
    def forward(self, x_q, x_k, x_v, mask=None):
        # x_q: (B, N_q, d_model)
        # x_k: (B, N_kv, d_model)
        # x_v: (B, N_kv, d_model)
        # mask: (B, N_q, N_kv)
        assert x_k.shape[-2] == x_v.shape[-2]
        q = self.q_linear(x_q) # (B, N_q, d_k)
        k = self.k_linear(x_k) # (B, N_kv, d_k)
        v = self.v_linear(x_v) # (B, N_kv, d_k)
        scaler = torch.sqrt(torch.tensor(self.d_k))
        score = (q @ k.transpose(1, 2)) / scaler # (B, N_q, N_kv)
        if mask is not None:
            assert mask.shape == score.shape
            score = self.mask(score, mask) # (B, N_q, N_kv)
        y = torch.softmax(score, dim=1) @ v # (B, N_q, d_k)
        return y
    
    def mask(self, score, mask):
        mask = torch.where(mask == 0, -1.e9, 0.)
        score += mask
        return score
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = int(d_model / num_heads)
        self.heads = nn.ModuleList([Attention(d_model=d_model, d_k=self.d_k) for _ in range(num_heads)])
        self.linear = nn.Linear(in_features=d_model, out_features=d_model)

    def forward(self, x_q, x_k, x_v, mask=None):
        # x_q: (B, N_q, d_model)
        # x_k: (B, N_kv, d_model)
        # x_v: (B, N_kv, d_model)
        # mask: (B, N_q, N_kv)
        outputs = []
        for head in self.heads:
            out = head(x_q, x_k, x_v, mask) # (B, N_q, d_k)
            outputs.append(out)
        y = torch.cat(outputs, dim=2) # (B, N_q, d_model)
        y = self.linear(y) # (B, N_q, d_model)
        return y
    
class MultiHeadSelfAttention(MultiHeadAttention):
    def forward(self, x, mask=None):
        y = super().forward(x, x, x, mask)
        return y
    
class FFNN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_ff = d_ff
        self.linear1 = nn.Linear(in_features=d_model, out_features=d_ff)
        self.linear2 = nn.Linear(in_features=d_ff, out_features=d_model)

    def forward(self, x):
        y = self.linear1(x)
        y = F.relu(y)
        y = self.linear2(y)
        return y

class Encoder(nn.Module):
    def __init__(self, num_heads, d_model, d_ff, dropout_rate=0.):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.self_attention = MultiHeadSelfAttention(num_heads=num_heads, d_model=d_model)
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.attention_layer_norm = nn.LayerNorm(d_model)
        self.ffnn = FFNN(d_model=d_model, d_ff=d_ff)
        self.ffnn_dropout = nn.Dropout(dropout_rate)
        self.ffnn_layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x, binary_mask=None):
        y = self.forward_self_attention(x, binary_mask)
        y = self.forward_ffnn(y)
        return y
    
    def forward_self_attention(self, x, binary_mask):
        mask = binary_mask_to_attention_mask(binary_mask, query_sequence_length=x.shape[1]) if binary_mask is not None else None
        y = self.self_attention(x, mask)
        y = self.attention_dropout(y)
        y = self.attention_layer_norm(x + y)
        return y
    
    def forward_ffnn(self, x):
        y = self.ffnn(x)
        y = self.ffnn_dropout(y)
        y = self.ffnn_layer_norm(x + y)
        return y

class GPTDecoder(Encoder):
    # TODO: implement gpt decoder with causal mask
    pass

class Decoder(Encoder):
    def __init__(self, num_heads, d_model, d_ff, dropout_rate=0.):
        super().__init__(num_heads, d_model, d_ff, dropout_rate)
        self.enc_dec_attention = MultiHeadAttention(num_heads=num_heads, d_model=d_model)
        self.enc_dec_attention_dropout = nn.Dropout(dropout_rate)
        self.enc_dec_attention_layer_norm = nn.LayerNorm(d_model)
  
    def forward(self, x, enc_y, enc_binary_mask=None, dec_binary_mask=None):
        y = self.forward_self_attention(x, dec_binary_mask)
        y = self.forward_enc_dec_attention(y, enc_y, enc_binary_mask)
        y = self.forward_ffnn(y)
        return y
    
    def forward_self_attention(self, x, dec_binary_mask):
        lookahead_mask = create_lookahead_mask(x)
        if dec_binary_mask is not None:
            dec_padding_mask = binary_mask_to_attention_mask(dec_binary_mask, query_sequence_length=x.shape[1]) if dec_binary_mask is not None else None
            mask = torch.maximum(dec_padding_mask, lookahead_mask)
        else:
            mask = lookahead_mask
        y = self.self_attention(x, mask)
        y = self.attention_dropout(y)
        y = self.attention_layer_norm(x + y)
        return y
    
    def forward_enc_dec_attention(self, x, enc_y, enc_binary_mask):
        mask = binary_mask_to_attention_mask(enc_binary_mask, query_sequence_length=x.shape[1]) if enc_binary_mask is not None else None
        y = self.enc_dec_attention(x, enc_y, enc_y, mask)
        y = self.enc_dec_attention_dropout(y)
        y = self.enc_dec_attention_layer_norm(y + x)
        return y
    
class PositionalEmbedding(nn.Module):
    def __init__(self, num_embeddings, d_embed):
        super().__init__()
        self.d_embed = d_embed
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=d_embed)
    
    def forward(self, x):
        # x: (B, N)
        y = self.embedding(x) # y: (B, N, d_embed)
        b, n = x.shape[0], x.shape[1]
        encoding = self.positional_encoding(n).unsqueeze(0).repeat(b, 1, 1)
        y += encoding.to(y)
        return y
    
    def positional_encoding(self, n):
        pos = torch.arange(n)# (n,)
        two_i = 2. * (torch.arange(self.d_embed) // 2.) # (d_embed,), [0., 0., 2., 2., 4., ...]
        angles = pos[:, None] / torch.pow(10000, (two_i / self.d_embed)) # (n, d_embed)
        encoding = angles.clone()
        encoding[:, 0::2] = torch.sin(angles[:, 0::2])
        encoding[:, 1::2] = torch.cos(angles[:, 1::2])
        return encoding
    
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_embed, d_model, num_layers, num_heads, d_ff, dropout_rate=0.):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_embed = d_embed
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.embedding = PositionalEmbedding(num_embeddings=vocab_size, d_embed=d_embed)
        self.dropout = nn.Dropout(dropout_rate)
        self.model_in_linear = nn.Linear(in_features=d_embed, out_features=d_model)
        self.encoders = nn.ModuleList([Encoder(num_heads=num_heads, d_model=d_model, d_ff=d_ff, dropout_rate=dropout_rate) for _ in range(num_layers)])
        self.decoders = nn.ModuleList([Decoder(num_heads=num_heads, d_model=d_model, d_ff=d_ff, dropout_rate=dropout_rate) for _ in range(num_layers)])
        self.head = nn.Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, enc_x, dec_x, enc_x_binary_mask=None, dec_x_binary_mask=None):
        enc_y = self.forward_encoders(enc_x, enc_x_binary_mask)
        dec_y = self.forward_decoders(dec_x, enc_y, enc_x_binary_mask, dec_x_binary_mask)
        y = self.head(dec_y)
        return y
    
    def forward_embedding(self, x):
        y = self.embedding(x)
        y = self.dropout(y)
        y = self.model_in_linear(y)
        return y
    
    def forward_encoders(self, enc_x, binary_mask):
        y = self.forward_embedding(enc_x)
        for encoder in self.encoders:
            y = encoder(y, binary_mask)
        return y
    
    def forward_decoders(self, dec_x, enc_y, enc_binary_mask, dec_binary_mask):
        y = self.forward_embedding(dec_x)
        for decoder in self.decoders:
            y = decoder(y, enc_y, enc_binary_mask, dec_binary_mask)
        return y
    
class GPT(nn.Module):
    def __init__(self, vocab_size, d_embed, d_model, num_decoders, num_heads, d_ff, dropout_rate=0.):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_embed = d_embed
        self.d_model = d_model
        self.num_decoders = num_decoders
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.embedding = PositionalEmbedding(num_embeddings=vocab_size, d_embed=d_embed)
        self.dropout = nn.Dropout(dropout_rate)
        self.model_in_linear = nn.Linear(in_features=d_embed, out_features=d_model)
        self.decoders = nn.ModuleList([GPTDecoder(num_heads=num_heads, d_model=d_model, d_ff=d_ff, dropout_rate=dropout_rate) for _ in range(num_decoders)])
        self.head = nn.Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, x, binary_mask=None):
        y = self.embedding(x)
        y = self.dropout(y)
        y = self.model_in_linear(y)
        for decoder in self.decoders:
            y = decoder(y, binary_mask)
        y = self.head(y) # (B, N, vocab_size)
        return y