import pytorch_lightning as pl
import torch


class Transformer(pl.LightningModule):
    def __init__(
        self,
        n_tokens,
        d_model,
        n_blocks,
        seq_length,
        block_dropout=0,
        n_classes=2,
        max_pool=False,
        padding_ix=0,
    ):

        super().__init__()

        self.embed = torch.nn.Embedding(
            num_embeddings=n_tokens, embedding_dim=d_model, padding_idx=0
        )

        self.pos = torch.nn.Embedding(seq_length, d_model)

        blocks = [
            DecoderLayer(d_model, n_heads=8, dropout=block_dropout)
            for _ in range(n_blocks)
        ]
        self.layers = torch.nn.Sequential(*blocks)

        self.norm = torch.nn.LayerNorm(d_model)
        self.to_scores = torch.nn.Linear(d_model, n_classes)
        self._max_pool = max_pool
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=padding_ix)

    def forward(self, x):
        tokens = self.embed(x)
        batch_size, seq_length, emb_dim = tokens.size()
        positions = torch.arange(seq_length, device=next(self.parameters()).device)
        positions = self.pos(positions)[None, :, :].expand(
            batch_size, seq_length, emb_dim
        )

        x = tokens + positions

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = x.max(dim=1)[0] if self._max_pool else x.mean(dim=1)
        return self.to_scores(x)

    def training_step(self, batch, batch_idx):
        # unclear why the example wants a batch index
        # still need to implement positional encoding
        # which would make use of the first item in "batch"
        _, x, y = batch
        # lightning recommends keeping the training logic separate
        # from the inference logic
        y_hat = self(x)
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
