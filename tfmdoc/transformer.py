import math

import pytorch_lightning as pl
import torch


class Transformer(pl.LightningModule):
    def __init__(
        self,
        n_tokens,
        d_model,
        n_blocks,
        max_len=6000,
        block_dropout=0,
        n_classes=2,
        max_pool=False,
        padding_ix=0,
    ):

        super().__init__()

        self.embed = torch.nn.Embedding(
            num_embeddings=n_tokens, embedding_dim=d_model, padding_idx=0
        )

        self.pos_encode = PositionalEncoding(d_model=d_model, max_len=max_len)

        blocks = [
            DecoderLayer(d_model, n_heads=8, dropout=block_dropout)
            for _ in range(n_blocks)
        ]
        self.layers = torch.nn.Sequential(*blocks)

        self.norm = torch.nn.LayerNorm(d_model)
        self.to_scores = torch.nn.Linear(d_model, n_classes)
        self._max_pool = max_pool
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=padding_ix)

    def forward(self, timestamps, codes):
        tokens = self.embed(codes)
        x = self.pos_encode(timestamps, tokens)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = x.max(dim=1)[0] if self._max_pool else x.mean(dim=1)
        return self.to_scores(x)

    def training_step(self, batch, batch_idx):
        # unclear why the example wants a batch index
        # still need to implement positional encoding
        # which would make use of the first item in "batch"
        t, x, y = batch
        # lightning recommends keeping the training logic separate
        # from the inference logic
        y_hat = self(t, x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class DecoderLayer(torch.nn.Module):
    def __init__(self, size, n_heads, dropout):

        super().__init__()
        self.self_attn = torch.nn.MultiheadAttention(size, n_heads)
        self.norm1 = torch.nn.LayerNorm(size)
        self.norm2 = torch.nn.LayerNorm(size)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.ff = torch.nn.Sequential(
            torch.nn.Linear(size, size * 4),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(size * 4, size),
        )

    def forward(self, x):
        attn = self.self_attn(x, x, x, need_weights=False)[0]
        x = x + self.dropout1(attn)
        x = self.norm1(x)
        fedfwd = self.ff(x)
        x = x + self.dropout2(fedfwd)
        return self.norm2(x)


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=6000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model),
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, t, x):
        x = x + self.pe[t]
        return self.dropout(x)
