import torch
import torch.nn as nn

class Recurrent(nn.Module):
    def __init__(self, size):
        super(Recurrent, self).__init__()
        self.size = size
        self.rnn = nn.RNN(size, size, num_layers=3, batch_first=True)
        self.fc = nn.Linear(size, size)

    def forward(self, x):
        x, _ = self.rnn(x)
        out = self.fc(x)
        out = torch.round(out)
        return out