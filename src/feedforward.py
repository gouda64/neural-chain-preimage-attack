import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()

        self.d1 = nn.Linear(432, 432)
        self.d2 = nn.Linear(432, 432)
        self.d3 = nn.Linear(432, 432)

    def forward(self, x):
        x = self.d1(x)
        x = F.relu(x)

        x = self.d2(x)
        x = F.relu(x)

        logits = self.d3(x)
        out = F.softmax(logits, dim=1)
        return out
