import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import data
import feedforward

learning_rate = 0.001
num_epochs = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = feedforward.FeedForward()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

## compute accuracy
def get_accuracy(logit, target, batch_size):
    ''' Obtain accuracy for training round '''
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()

for epoch in range(num_epochs):
    train_running_loss = 0.0
    train_acc = 0.0

    model = model.train()

    ## training step
    for i, (layers, prevLayers) in enumerate(data.train_loader):
        layers = layers.to(device)
        prevLayers = prevLayers.to(device)

        ## forward + backprop + loss
        logits = model(layers)
        loss = criterion(logits, prevLayers)
        optimizer.zero_grad()
        loss.backward()

        ## update model params
        optimizer.step()

        train_running_loss += loss.detach().item()
        train_acc += get_accuracy(logits, prevLayers, data.BATCH_SIZE)

    model.eval()
    test_acc = 0.0
    for i, (layers, prevLayers) in enumerate(data.test_loader, 0):
        layers = layers.to(device)
        prevLayers = prevLayers.to(device)
        outputs = model(layers)
        test_acc += get_accuracy(outputs, prevLayers, data.BATCH_SIZE)

    print('Epoch: %d | Loss: %.4f | Train Accuracy: %.2f' \
          %(epoch, train_running_loss / i, train_acc/i))
    print('Test Accuracy: %.2f'%( test_acc/i))


torch.save(model, "/model.pt")