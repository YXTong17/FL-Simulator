import torch
from .client import Client


class fedfgac(Client):
    def __init__(self, device, model_func, received_vecs, dataset, lr, args):
        super().__init__(device, model_func, received_vecs, dataset, lr, args)
        # Extract unique class labels from the dataset
        labels = torch.tensor(dataset[1]).squeeze()
        self.label_counts = torch.bincount(labels, minlength=self.model.n_cls)
        self.label_counts = torch.log(1 + self.label_counts)

    def train(self):
        super().train()
        self.comm_vecs["local_label_counts"] = self.label_counts
        return self.comm_vecs
