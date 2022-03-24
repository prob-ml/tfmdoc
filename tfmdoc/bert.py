import math

import pytorch_lightning as pl
import torch
from torch.nn import Linear


class BERT(pl.LightningModule):
    def __init__(
        self,
        n_tokens,
        d_model,
        n_blocks,
        n_heads,
        dropout,
        lr,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.embed = torch.nn.Embedding(
            num_embeddings=n_tokens, embedding_dim=d_model, padding_idx=0
        )
        self.age_embed = torch.nn.Embedding(
            # assume a max age of 100
            num_embeddings=100,
            embedding_dim=d_model,
            padding_idx=0,
        )
        self.results = None
        self.pos_encode = PositionalEncoding(d_model=d_model)
        blocks = [
            DecoderLayer(d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(n_blocks)
        ]
        self._norm = torch.nn.LayerNorm(d_model)
        self._layers = torch.nn.Sequential(*blocks)
        self._to_scores = Linear(d_model, n_tokens)
        self._loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

        self._n_heads = n_heads

    def forward(self, ages, visits, codes):
        # embed codes into dimension of model
        # for continuous representation
        mask = (codes < 1).unsqueeze(1).repeat(1, codes.size(1), 1)
        # very helpful documentation! (not)
        mask = mask.repeat_interleave(repeats=self._n_heads, dim=0)
        x = self.embed(codes)
        # apply position encodings
        x = self.pos_encode(visits, x)
        x = x + self.age_embed(ages)
        # wat happens if an entire batch is masked out??
        for layer in self._layers:
            x = layer(x, mask)

        x = self._norm(x)
        # shape will be (n_batches, d_model)
        # final linear layer projects this down to (n_batches, n_classes
        return self._to_scores(x)

    def training_step(self, batch, batch_idx):
        t, v, _, x, y = batch
        y_hat = self(t, v, x)
        loss = self._loss_fn(y_hat.transpose(1, 2), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        t, v, _, x, y = batch
        y_hat = self(t, v, x)
        loss = self._loss_fn(y_hat.transpose(1, 2), y)
        self.log("val_loss", loss)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=1e-4
        )


class DecoderLayer(torch.nn.Module):
    def __init__(self, size, n_heads, dropout):

        super().__init__()
        self.self_attn = torch.nn.MultiheadAttention(size, n_heads, batch_first=True)
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

    def forward(self, x, mask=None):
        attn = self.self_attn(x, x, x, need_weights=False, attn_mask=mask)[0]
        x = x + self.dropout1(attn)
        x = self.norm1(x)
        fedfwd = self.ff(x)
        x = x + self.dropout2(fedfwd)
        return self.norm2(x)


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model),
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # non-optimizable parameter
        self.register_buffer("pe", pe)

    def forward(self, v, x):
        # x has shape (n_batches, seq_length, d_model)
        # expand position encodings along batch dimension
        pos = self.pe.expand(v.shape[0], self.pe.shape[1], self.pe.shape[2])
        # expand visit indices along embedding dimension
        v = v.unsqueeze(2).expand(v.shape[0], v.shape[1], pos.shape[2])
        # select visit encodings for each patient
        pos_encoding = pos.gather(1, v.long())
        # add encodings to embedded input
        x += pos_encoding
        return self.dropout(x)
