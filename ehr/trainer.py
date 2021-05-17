import torch


class Trainer:
    def __init__(self, model, batch_size, optimizer, padding_ix=0):

        self.model = model
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.padding_ix = padding_ix

    def train(self, training_data, epochs):

        if torch.cuda.is_available():
            self.model.cuda()

        training_losses = []
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.padding_ix)
        features, labels = training_data
        features = torch.from_numpy(features)
        labels = torch.from_numpy(labels)
        if torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()
        n_sequences = features.shape[0]
        for _ in range(epochs):
            for i in range(0, n_sequences, self.batch_size):
                if i + self.batch_size > n_sequences:
                    continue
                x = features[i : i + self.batch_size, :]
                y = labels[i : i + self.batch_size]
                y_hat = self.model(x)
                loss = loss_fn(y_hat, y)

                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                training_losses.append(loss.item())

        return training_losses

    def __str__(self):
        # placeholder until we start using lightning
        return self.__class__.__name__
