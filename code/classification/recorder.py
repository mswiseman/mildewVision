import torch


class Recorder:
    def __init__(self, name):
        self._name = name
        self.reset()

    def update(self, logits, labels, loss, batch_size=None):
        """
        logits: (B, C)
        labels: (B,)
        loss: float (usually mean loss for the batch)
        """
        if batch_size is None:
            batch_size = labels.size(0)

        self.correct_counts += self.count_correct(logits, labels)
        self.total_counts += batch_size

        # store total summed loss over all samples
        self.loss_sum += float(loss) * batch_size

    @property
    def name(self):
        return self._name

    @property
    def loss(self):
        # mean loss per sample
        return self.loss_sum / max(self.total_counts, 1)

    @property
    def correct(self):
        return self.correct_counts

    @property
    def total(self):
        return self.total_counts

    def reset(self):
        self.loss_sum = 0.0
        self.correct_counts = 0
        self.total_counts = 0

    def count_correct(self, logits, labels):
        preds = torch.argmax(logits, dim=1)
        return (preds == labels).sum().item()