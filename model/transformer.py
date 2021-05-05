from torch import nn

from helpers.utils import clones


class Transformer(nn.Module):
    def __init__(
        self, n_tokens, d_model, n_blocks, seq_length, block_dropout=0, n_classes=2
    ):

        super().__init__()

        self.embed = nn.Embedding(
            num_embeddings=n_tokens, embedding_dim=d_model, padding_idx=0
        )

        self.pos = nn.Embedding(seq_length, d_model)

        self.layers = clones(
            DecoderLayer(d_model, n_heads=8, dropout=block_dropout), n_copies=n_blocks
        )

        self.norm = nn.LayerNorm(d_model)
        self.to_scores = nn.Linear(d_model, n_classes)

    def forward(self, x):

        x = self.embed(x)
        t = x.size()[1]

        positions = self.pos[:, :t, :]
        x += positions

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return self.to_scores(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, n_heads, dropout):

        super().__init__()
        self.self_attn = nn.MultiheadAttention(size, n_heads)
        self.norm1 = nn.LayerNorm(size)
        self.norm2 = nn.LayerNorm(size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(size, size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(size * 4, size),
        )

    def forward(self, x):
        attn = self.self_attn(x)
        x = x + self.dropout1(attn)
        x = self.norm1(x)
        fedfwd = self.ff(x)
        x = x + self.dropout2(fedfwd)
        return self.norm2(x)
