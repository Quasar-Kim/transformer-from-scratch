import torch
from torch import nn
import torch.nn.functional as F

def create_lookahead_mask(x: torch.Tensor):
    # x: (B, N, d_model)
    assert len(x.shape) == 3
    b, n, _ = x.shape
    mask = torch.tril(torch.ones(n, n, device=x.device))
    mask = mask.unsqueeze(0).repeat(b, 1, 1)
    return mask

def binary_mask_to_attention_mask(key_binary_mask: torch.Tensor, query_sequence_length: int):
    # binary_mask: (B, N_kv)
    # mask: (B, N_q, N_kv)
    mask = key_binary_mask.float().unsqueeze(1).repeat(1, query_sequence_length, 1)
    return mask
    
class MultiHeadAttention(nn.Module):
    def __init__(self, *, num_heads, d_model):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads # = d_model / n_heads
        self.q_linear = nn.Linear(in_features=d_model, out_features=d_model)
        self.k_linear = nn.Linear(in_features=d_model, out_features=d_model)
        self.v_linear = nn.Linear(in_features=d_model, out_features=d_model)
        self.dense = nn.Linear(in_features=self.d_model, out_features=self.d_model)
    
    def forward(self, x_q, x_k, x_v, mask=None):
        # x_q: (B, N_q, d_model)
        # x_k: (B, N_kv, d_model)
        # x_v: (B, N_kv, d_model)
        # mask: (B, N_q, N_kv)
        assert x_k.shape == x_v.shape
        b, n_q, _ = x_q.shape
        q = self.q_linear(x_q) # (B, N_q, d_model)
        q = self.split_head(q) # (B, heads, N_q, d_k)
        k = self.k_linear(x_k) # (B, N_kv, d_model)
        k = self.split_head(k) # (B, heads, N_kv, d_k)
        v = self.v_linear(x_v) # (B, N_kv, d_model)
        v = self.split_head(v) # (B, heads, N_kv, d_k)
        scaler = torch.sqrt(torch.tensor(float(self.d_k), device=x_q.device))
        logits = (q @ k.transpose(2, 3)) / scaler # (B, heads, N_q, N_kv)
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1) # (B, heads, N_q, N_kv)
            assert mask.shape == logits.shape
            logits = self.mask(logits, mask)
        score = torch.softmax(logits, dim=3) # (B, heads, N_q, N_kv)
        attn = score @ v # (B, heads, N_q, d_k)
        attn = attn.transpose(1, 2).reshape((b, n_q, self.d_model)) # (B, N_q, d_model)
        y = self.dense(attn) # (B, N_q, d_model)
        return y
    
    def split_head(self, x):
        # x: (B, N, d_model)
        b, n, _ = x.shape
        y = x.reshape((b, n, self.num_heads, self.d_k)) # (B, N, num_heads, d_k)
        y = y.transpose(1, 2) # (B, num_heads, N, d_k)
        return y
    
    def mask(self, score, mask):
        mask = torch.where(mask == 0, -1.e9, 0.)
        score += mask
        return score
    
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
        self.attention_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.ffnn = FFNN(d_model=d_model, d_ff=d_ff)
        self.ffnn_dropout = nn.Dropout(dropout_rate)
        self.ffnn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
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

class Decoder(nn.Module):
    def __init__(self, num_heads, d_model, d_ff, dropout_rate=0.):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.self_attention = MultiHeadSelfAttention(num_heads=num_heads, d_model=d_model)
        self.self_attention_dropout = nn.Dropout(dropout_rate)
        self.self_attention_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.enc_dec_attention = MultiHeadAttention(num_heads=num_heads, d_model=d_model)
        self.enc_dec_attention_dropout = nn.Dropout(dropout_rate)
        self.enc_dec_attention_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.ffnn = FFNN(d_model=d_model, d_ff=d_ff)
        self.ffnn_dropout = nn.Dropout(dropout_rate)
        self.ffnn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
  
    def forward(self, x, enc_y, enc_binary_mask=None, dec_binary_mask=None):
        y = self.forward_self_attention(x, dec_binary_mask)
        y = self.forward_enc_dec_attention(y, enc_y, enc_binary_mask)
        y = self.forward_ffnn(y)
        return y
    
    def forward_self_attention(self, x, dec_binary_mask):
        lookahead_mask = create_lookahead_mask(x)
        if dec_binary_mask is not None:
            dec_padding_mask = binary_mask_to_attention_mask(dec_binary_mask, query_sequence_length=x.shape[1]) if dec_binary_mask is not None else None
            mask = torch.minimum(dec_padding_mask, lookahead_mask)
        else:
            mask = lookahead_mask
        y = self.self_attention(x, mask)
        y = self.self_attention_dropout(y)
        y = self.self_attention_layer_norm(x + y)
        return y
    
    def forward_enc_dec_attention(self, x, enc_y, enc_binary_mask):
        mask = binary_mask_to_attention_mask(enc_binary_mask, query_sequence_length=x.shape[1]) if enc_binary_mask is not None else None
        y = self.enc_dec_attention(x, enc_y, enc_y, mask)
        y = self.enc_dec_attention_dropout(y)
        y = self.enc_dec_attention_layer_norm(y + x)
        return y
    
    def forward_ffnn(self, x):
        y = self.ffnn(x)
        y = self.ffnn_dropout(y)
        y = self.ffnn_layer_norm(x + y)
        return y
    
class PositionalEmbedding(nn.Module):
    def __init__(self, num_embeddings, d_embed):
        super().__init__()
        self.d_embed = d_embed
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=d_embed)
    
    def forward(self, x):
        # x: (B, N)
        y = self.embedding(x) # y: (B, N, d_embed)
        # why? - gradient vanishing 문제를 완화나는 테크닉이라고 함, implementation detail인듯
        y *= torch.sqrt(torch.tensor(float(self.d_embed)))
        y = self.encode_position(y)
        return y

    def encode_position(self, y: torch.Tensor):
        b, n, _ = y.shape
        pos = torch.arange(n, device=y.device).unsqueeze(1).repeat(1, self.d_embed) # (N, d_embed)
        two_i = (2. * (torch.arange(self.d_embed, device=y.device) // 2.)).unsqueeze(0).repeat(n, 1) # (N, d_embed)
        angles = pos / torch.pow(10000., (two_i / self.d_embed)) # (N, d_embed)
        encoding = torch.empty((n, self.d_embed), device=y.device)
        indices = torch.tensor([True, False], device=y.device).repeat((self.d_embed // 2) + 1)[:self.d_embed]
        encoding = torch.where(indices, torch.sin(angles), torch.cos(angles))
        encoding = encoding.unsqueeze(0).repeat(b, 1, 1)
        y += encoding
        return y
    
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