import time

import torch


class Trainer:
    def __init__(self, model, batch_size, optimizer):

        self.model = model
        self.batch_size = batch_size
        self.optimizer = optimizer

    def train(self, training_data, epochs):
        training_start_time = time.time()

        if torch.cuda.is_available():
            self.model.cuda()

        training_losses = []
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
                X = features[i:i + 16, :]
                y = labels[i:i + 16]
                y_hat = self.model(X)
