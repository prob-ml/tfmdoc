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
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=d_model * 4,
        )
        norm = torch.nn.LayerNorm(d_model)
        self._transformer = torch.nn.TransformerEncoder(encoder_layer, n_blocks, norm)
        self._to_scores = Linear(d_model, n_tokens)
        self._loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

        self._n_heads = n_heads

    def forward(self, ages, visits, codes):
        # embed codes into dimension of model
        # for continuous representation
        mask = (codes < 1).unsqueeze(1).repeat(1, codes.size(1), 1)
        # very helpful documentation! (not)
        mask = mask.repeat_interleave(repeats=self._n_heads, dim=0)
        pad_mask = codes == 0
        x = self.embed(codes)
        # apply position encodings
        x = self.pos_encode(visits, x)
        x = x + self.age_embed(ages)
        # wat happens if an entire batch is masked out??
        x = self._transformer(x, mask, pad_mask)
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
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


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
