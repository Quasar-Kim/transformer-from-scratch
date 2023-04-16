import torch
from torchview import draw_graph
from modules import Transformer

enc_n = 4
dec_n = 8
vocab_size = 2**14
d_model = 6
d_embed = 6
num_decoders = 1
num_heads = 2
d_ff = 16
module = Transformer(vocab_size=vocab_size, d_embed=d_embed, d_model=d_model, num_layers=num_decoders, num_heads=num_heads, d_ff=d_ff, dropout_rate=0.1)
module.eval()

enc_x1 = torch.randint(0, vocab_size, (1, enc_n))
dec_x1 = torch.randint(0, vocab_size, (1, dec_n))

graph = draw_graph(module, input_data=[enc_x1, dec_x1], graph_name='transformer', expand_nested=True, save_graph=True)
graph.visual_graph.render(format='svg')