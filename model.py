import torch 
import torch.nn as nn

class SwitchTransformer(nn.Module):
    def __init__(self, num_layers, num_heads, d_model, d_ff, num_switch, num_switch_layer, vocab_size, dropout=0.0):
        super(SwitchTransformer, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_switch = num_switch
        self.num_switch_layer = num_switch_layer
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(1000, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(SwitchTransformerLayer(num_heads, d_model, d_ff, num_switch, num_switch_layer, dropout))

        self.out = nn.Linear(d_model, vocab_size)


    def forward(self, x, y):
        # x: (batch_size, seq_len)
        # y: (batch_size, seq_len)
        batch_size, seq_len = x.size()
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        # pos: (batch_size, seq_len)
        x = self.embed(x) + self.pos_embed(pos)
        # x: (batch_size, seq_len, d_model)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, y)
        # x: (batch_size, seq_len, d_model)
        x = self.out(x)
        # x: (batch_size, seq_len, vocab_size)
        # get logprobs
        x = nn.functional.log_softmax(x, dim=-1)
        # x: (batch_size, seq_len, vocab_size)
        return x
    
class SwitchTransformerLayer(nn.Module):
    def __init__(self, num_heads, d_model, d_ff, num_switch, num_switch_layer, dropout=0.0):
        super(SwitchTransformerLayer, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_switch = num_switch
        self.num_switch_layer = num_switch_layer
        self.dropout = dropout

        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.switch = nn.ModuleList()
        for _ in range(num_switch_layer):
            self.switch.append(nn.Linear(d_model, num_switch))
        self.switch_out = nn.Linear(num_switch, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, y):
        # x: (batch_size, seq_len, d_model)
        # y: (batch_size, seq_len)
        residual = x
        x, _ = self.self_attn(x, x, x)
        x = self.norm1(x + residual)
        # x: (batch_size, seq_len, d_model)

        residual = x
        x = self.ff(x)
        x = self.norm2(x + residual)
        # x: (batch_size, seq_len, d_model)

        residual = x
        x = self.switch[0](x)
        x = self.switch_out(x)
        x = self.norm3(x + residual)
        # x: (batch_size, seq_len, d_model)
        return x