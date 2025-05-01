import torch
from torch import nn


class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y) ** 2 + self.epsilon ** 2))