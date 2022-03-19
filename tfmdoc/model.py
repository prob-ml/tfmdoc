import math

import pytorch_lightning as pl
import torch
import torchmetrics
from torch.nn import Dropout, Linear, ReLU
from torch.nn.functional import relu, softmax


class Tfmd(pl.LightningModule):
    def __init__(
        self,
        n_tokens,
        d_model,
        n_blocks,
        n_heads,
        dropout,
        d_ff,
        transformer,
        d_bow,
        lr,
        mask=False,
    ):
        """Deep learning model for early detection of disease based on
            health insurance claims data. This model makes use of both
            patient records and demographic data to make predictions.
            The main model is a transformer, with a bag-of-words neural network
            serving as a performance baseline.

        Args:
            n_tokens (integer): number of unique tokens (codes) in features
            d_model (integer): dimension of each (transformer) decoder layer
            n_blocks (integer): number of stacked transformer blocks
            n_heads (integer): number of attention heads
            dropout (float): proportion of nodes to randomly zero during droupout
            d_ff (integer): dimension of feedforward layer that projects
                output of transformer down to 2 (number of classes to predict)
            transformer (bool): If true, use the transformer architecture.
                If false, architecture is a bag-of-words feedforward NN.
            d_bow (integer): dimension(s) of FF layer(s) in bag-of-words
                model.
            lr (float): learning rate
        """  # noqa: RST301
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
        if transformer:
            self.pos_encode = PositionalEncoding(d_model=d_model)
            blocks = [
                DecoderLayer(d_model, n_heads=n_heads, dropout=dropout)
                for _ in range(n_blocks)
            ]
            self._norm = torch.nn.LayerNorm(d_model)
            self._layers = torch.nn.Sequential(*blocks)
            self._final = Linear(d_model + 2, d_ff)
        else:
            d_bow.append(d_model)
            bow_layers = make_bow_layers(n_tokens, d_bow, dropout=dropout)
            self.dense = torch.nn.Sequential(*bow_layers)
            self._final = Linear(d_bow[-1] + 2, d_ff)
        self.pr_curve = torchmetrics.PrecisionRecallCurve(pos_label=1)
        # add two to input dim for demographic data (sex, age)
        self._to_scores = Linear(d_ff, 2)
        self._loss_fn = torch.nn.CrossEntropyLoss()
        self._accuracy = torchmetrics.Accuracy()
        self._val_auroc = torchmetrics.AUROC(compute_on_step=False)
        self._test_auroc = torchmetrics.AUROC(compute_on_step=False)
        self._n_heads = n_heads
        self._mask = mask

    def forward(self, ages, visits, demo, codes):
        # embed codes into dimension of model
        # for continuous representation
        if self.hparams.transformer:
            if self._mask:
                mask = (codes > 0).unsqueeze(1).repeat(1, codes.size(1), 1)
                # very helpful documentation! (not)
                mask = mask.repeat_interleave(repeats=self._n_head, dim=0)
            else:
                mask = None
            x = self.embed(codes)
            # apply position encodings
            x = self.pos_encode(visits, x)
            x = x + self.age_embed(ages)
            for layer in self._layers:
                x = layer(x, mask)

            x = self._norm(x)
            x = x.max(dim=1)[0]
            # shape will be (n_batches, d_model)
            # final linear layer projects this down to (n_batches, n_classes)
        else:
            x = torch.cat((codes, ages), axis=1)
            x = self.dense(x)
        x = torch.cat((x, demo), axis=1)
        x = relu(self._final(x))
        return self._to_scores(x)

    def training_step(self, batch, batch_idx):
        t, v, w, x, y = batch
        y_hat = self(t, v, w, x)
        loss = self._loss_fn(y_hat, y)
        self.log("train_loss", loss)
        probas = softmax(y_hat, dim=1)[:, 1]
        acc = self._accuracy((probas > 0.5), y)
        self.log("train_accuracy", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        t, v, w, x, y = batch
        y_hat = self(t, v, w, x)
        loss = self._loss_fn(y_hat, y)
        self.log("val_loss", loss)
        probas = softmax(y_hat, dim=1)[:, 1]
        acc = self._accuracy((probas > 0.5), y)
        self.log("val_accuracy", acc)
        self._val_auroc(probas, y)
        self.log("val_auroc", self._val_auroc.compute(), on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        t, v, w, x, y = batch
        y_hat = self(t, v, w, x)
        probas = softmax(y_hat, dim=1)[:, 1]
        self._test_auroc(probas, y)
        self.log("test_auroc", self._test_auroc.compute(), on_step=False, on_epoch=True)
        return probas, y

    def test_epoch_end(self, outputs):
        probas, targets = zip(*outputs)
        probas = torch.cat(probas)
        targets = torch.cat(targets)
        self.results = (probas, targets)

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


# helpers


def make_bow_layers(n_tokens, d_bow, dropout):
    # account for age embeddings
    l0 = n_tokens + 100
    layers = []
    for dim in d_bow:
        l1 = dim
        layers.append(Linear(l0, l1))
        layers.append(ReLU())
        l0 = dim
    layers.append(Dropout(dropout))
    return layers
