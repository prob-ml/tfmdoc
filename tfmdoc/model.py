import torch
import torchmetrics
from torch.nn import Dropout, Linear, ReLU
from torch.nn.functional import relu, softmax

from tfmdoc.bert import BERT


class Tfmd(BERT):
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
        d_demo,
        lr,
        bert=None,
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
        super().__init__(
            n_tokens,
            d_model,
            n_blocks,
            n_heads,
            dropout,
            lr,
        )
        if bert is not None:
            self.load_from_checkpoint(
                bert,
                d_ff=d_ff,
                d_demo=d_demo,
                transformer=True,
                d_bow=False,
                strict=False,
            )
        self.save_hyperparameters()
        self.results = None
        if transformer:
            self._final = Linear(d_model + d_demo, d_ff)
        else:
            d_bow.append(d_model)
            bow_layers = make_bow_layers(n_tokens, d_bow, dropout=dropout)
            self.dense = torch.nn.Sequential(*bow_layers)
            self._final = Linear(d_bow[-1] + d_demo, d_ff)
        self.pr_curve = torchmetrics.PrecisionRecallCurve(pos_label=1)
        self._to_score = Linear(d_ff, 2)
        self._loss_fn = torch.nn.CrossEntropyLoss()
        self._accuracy = torchmetrics.Accuracy()
        self._val_auroc = torchmetrics.AUROC(compute_on_step=False)
        self._test_auroc = torchmetrics.AUROC(compute_on_step=False)

    def forward(self, ages, visits, demo, codes):
        # embed codes into dimension of model
        # for continuous representation
        if self.hparams.transformer:
            x = self.embed(codes)
            # apply position encodings
            x = self.pos_encode(visits, x)
            x = x + self.age_embed(ages)
            mask = None
            pad_mask = codes == 0
            x = self._transformer(x, mask, pad_mask)
            x = x.max(dim=1)[0]
            # shape will be (n_batches, d_model)
            # final linear layer projects this down to (n_batches, n_classes)
        else:
            x = torch.cat((codes, ages), axis=1)
            x = self.dense(x)
        x = torch.cat((x, demo), axis=1)
        x = relu(self._final(x))
        return self._to_score(x)

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

    def predict_step(self, batch, batch_ix):
        # drop last element (true labels)
        # if they are defined
        batch = batch[:4]
        y_hat = self(*batch)
        return softmax(y_hat, dim=1)[:, 1]


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
