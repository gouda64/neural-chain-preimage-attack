import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, size):
        super(FeedForward, self).__init__()

        self.d1 = nn.Linear(size, size)
        self.d2 = nn.Linear(size, size)
        self.d3 = nn.Linear(size, size)
        # self.d4 = nn.Linear(size, size)
        # self.d5 = nn.Linear(size, size)
        # self.d6 = nn.Linear(size, size)
        # self.d7 = nn.Linear(size, size)

    def forward(self, x):
        x = self.d1(x)
        x = F.relu(x)

        x = self.d2(x)
        x = F.relu(x)

        # x = self.d3(x)
        # x = F.relu(x)
        #
        # x = self.d4(x)
        # x = F.relu(x)
        #
        # x = self.d5(x)
        # x = F.relu(x)
        #
        # x = self.d6(x)
        # x = F.relu(x)

        logits = self.d3(x)
        out = F.softmax(logits, dim=1)
        return out
