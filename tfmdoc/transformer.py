import math

import pytorch_lightning as pl
import torch
import torchmetrics
from torch.nn.functional import softmax


class Transformer(pl.LightningModule):
    def __init__(
        self,
        n_tokens,
        d_model,
        d_demo,
        n_blocks,
        n_heads,
        max_len,
        block_dropout,
        max_pool,
    ):

        super().__init__()

        self.embed = torch.nn.Embedding(
            num_embeddings=n_tokens, embedding_dim=d_model, padding_idx=0
        )

        self.pos_encode = PositionalEncoding(d_model=d_model, max_len=max_len)

        blocks = [
            DecoderLayer(d_model, n_heads=n_heads, dropout=block_dropout)
            for _ in range(n_blocks)
        ]
        self.layers = torch.nn.Sequential(*blocks)

        self.norm = torch.nn.LayerNorm(d_model)
        # binary classification
        self.to_scores = torch.nn.Linear(d_model + d_demo, 2)
        self._max_pool = max_pool
        self._loss_fn = torch.nn.CrossEntropyLoss()
        self._accuracy = torchmetrics.Accuracy()
        self._d_demo = d_demo
        self._auroc = torchmetrics.AUROC(pos_label=1)

    def forward(self, demo, codes):
        # embed codes into dimension of model
        # for continuous representation
        x = self.embed(codes)
        # add sinusoidal position encodings
        x = self.pos_encode(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = x.max(dim=1)[0] if self._max_pool else x.mean(dim=1)
        # shape will be (n_batches, d_model)
        if self._d_demo != 0:
            x = torch.cat((x, demo), axis=1)
        # final linear layer projects this down to (n_batches, n_classes)
        return self.to_scores(x)

    def training_step(self, batch, batch_idx):
        # unclear why the lightning api wants a batch index var
        w, x, y = batch
        # lightning recommends keeping the training logic separate
        # from the inference logic, though this works fine
        y_hat = self(w, x)
        loss = self._loss_fn(y_hat, y)
        self.log("train_loss", loss)
        probas = softmax(y_hat, dim=1)[:, 1]
        acc = self._accuracy((probas > 0.5), y)
        self.log("train_accuracy", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        # unclear why the lightning api wants a batch index var
        w, x, y = batch
        # lightning recommends keeping the training logic separate
        # from the inference logic, though this works fine
        y_hat = self(w, x)
        loss = self._loss_fn(y_hat, y)
        self.log("val_loss", loss)
        probas = softmax(y_hat, dim=1)[:, 1]
        acc = self._accuracy((probas > 0.5), y)
        self.log("val_accuracy", acc)
        auroc = self._auroc(probas, y)
        self.log("val_auroc", auroc)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)


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
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x has shape (n_batches, seq_length, d_model)
        # add encodings for positions found in batched sequences
        t = x.shape[1]
        x = x + self.pe[:, :t, :]
        return self.dropout(x)
