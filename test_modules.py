import torch

from modules import Attention, MultiHeadAttention, MultiHeadSelfAttention, create_lookahead_mask, binary_mask_to_attention_mask, FFNN, Encoder, Decoder, PositionalEmbedding, GPT, Transformer

class TestAttention:
    def test_shape(self):
        n = 4
        d_model = 10
        num_head = 2
        d_k = int(d_model / num_head)
        x = torch.rand((2, n, d_model))
        module = Attention(d_model=d_model, d_k=d_k)
        y = module(x, x, x)
        assert tuple(y.shape) == (2, n, d_k)

class TestMultiHeadAttention:
    def test_shape(self):
        n = 4
        d_model = 10
        num_heads = 2
        module = MultiHeadAttention(num_heads=num_heads, d_model=d_model)
        x = torch.rand((2, n, d_model))
        y = module(x, x, x)
        assert x.shape == y.shape

class TestMultiHeadSelfAttention:
    def test_shape(self):
        n = 4
        d_model = 10
        num_heads = 2
        module = MultiHeadSelfAttention(num_heads=num_heads, d_model=d_model)
        x = torch.rand((2, n, d_model))
        y = module(x)
        assert x.shape == y.shape

def test_create_lookahead_mask():
    x = torch.Tensor(1, 4, 512)
    mask = create_lookahead_mask(x)
    correct_mask = torch.tensor([[
        [1., 0., 0., 0.],
        [1., 1., 0., 0.],
        [1., 1., 1., 0.],
        [1., 1., 1., 1.],
    ]])
    assert torch.equal(mask, correct_mask)

def test_binary_mask_to_attention_mask():
    binary_mask = torch.tensor([[1, 1, 0, 0]])
    mask = binary_mask_to_attention_mask(binary_mask, 5)
    correct_mask = torch.tensor([[
        [1., 1., 0., 0.],
        [1., 1., 0., 0.],
        [1., 1., 0., 0.],
        [1., 1., 0., 0.],
        [1., 1., 0., 0.],
    ]])
    assert torch.equal(mask, correct_mask)

class TestFFNN:
    def test_shape(self):
        n = 4
        d_model = 10
        module = FFNN(d_model=d_model, d_ff=100)
        x = torch.rand((2, n, d_model))
        y = module(x)
        assert x.shape == y.shape

class TestEncoder:
    def test_shape(self):
        n = 4
        d_model = 10
        num_heads = 2
        d_ff = 100
        module = Encoder(num_heads=num_heads, d_model=d_model, d_ff=d_ff)
        x = torch.rand(2, n, d_model)
        y = module(x)
        assert x.shape == y.shape

class TestDecoder:
    def test_shape(self):
        n_enc = 4
        n_dec = 7
        d_model = 10
        num_heads = 2
        d_ff = 100
        module = Decoder(num_heads=num_heads, d_model=d_model, d_ff=d_ff)
        enc_y = torch.rand(2, n_enc, d_model)
        x = torch.rand(2, n_dec, d_model)
        y = module(x, enc_y)
        assert x.shape == y.shape

class TestPositionalEmbedding:
    def test_shape(self):
        n = 4
        vocab_size = 20
        d_embed = 100
        module = PositionalEmbedding(num_embeddings=vocab_size, d_embed=d_embed)
        x = torch.randint(0, 10, (2, n))
        y = module(x)
        assert tuple(y.shape) == (2, n, d_embed)
        
class TestGPT:
    def test_shape(self):
        n = 4
        vocab_size = 20
        d_model = 10
        d_embed = 100
        num_decoders = 6
        num_heads = 2
        d_ff = 100
        module = GPT(vocab_size=vocab_size, d_embed=d_embed, d_model=d_model, num_decoders=num_decoders, num_heads=num_heads, d_ff=d_ff)
        x = torch.randint(0, 10, (2, n))
        y = module(x)
        assert y.shape == (2, n, vocab_size)

class TestTransformer:
    def test_shape(self):
        enc_n = 4
        dec_n = 8
        vocab_size = 20
        d_model = 10
        d_embed = 100
        num_decoders = 6
        num_heads = 2
        d_ff = 100
        module = Transformer(vocab_size=vocab_size, d_embed=d_embed, d_model=d_model, num_layers=num_decoders, num_heads=num_heads, d_ff=d_ff)
        enc_x = torch.randint(0, 10, (2, enc_n))
        dec_x = torch.randint(0, 10, (2, dec_n))
        y = module(enc_x, dec_x)
        assert y.shape == (2, dec_n, vocab_size)