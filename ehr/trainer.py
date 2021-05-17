import time

import torch
from torch import nn


class Trainer:
    def __init__(self, model, batch_size, optimizer, padding_ix=0):

        self.model = model
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.padding_ix = padding_ix

    def train(self, training_data, epochs):
        training_start_time = time.time()

        if torch.cuda.is_available():
            self.model.cuda()

        training_losses = []
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.padding_ix)
        features, labels = training_data
        features = torch.from_numpy(features)
        labels = torch.from_numpy(labels)
        if torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()
        n_sequences = features.shape[0]
        for epoch in range(epochs):
            for i in range(0, n_sequences, self.batch_size):
                if i + self.batch_size > n_sequences:
                    continue
                X = features[i : i + self.batch_size, :]
                y = labels[i : i + self.batch_size]
                y_hat = self.model(X)
                loss = loss_fn(y_hat, y)

                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                training_loss = loss.item()
                print(training_loss)
                training_losses.append(training_loss)

        return training_losses
