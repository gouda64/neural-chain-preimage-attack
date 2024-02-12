import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import data
import feedforward
import recurrent

size = 160

learning_rate = 0.01
num_epochs = 11

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = recurrent(size)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def get_accuracy(logit, target, batch_size):
    corrects = (logit == target).all(axis=1).sum().item()
    return corrects/batch_size

def get_bit_accuracy(logit, target, batch_size):
    corrects = (logit == target).sum().item()
    return corrects/(batch_size*size)

def train(train_loader, test_loader, model):
    train_len = len(train_loader.dataset);
    test_len = len(test_loader.dataset)

    for epoch in range(num_epochs):
        train_running_loss = 0.0
        train_acc = 0.0

        model = model.train()
        correct = 0

        for i, (layers, prevLayers) in enumerate(train_loader):
            layers = layers.to(device)
            prevLayers = prevLayers.to(device)

            logits = model(layers)
            loss = criterion(logits, prevLayers)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            train_running_loss += loss.detach().item()
            train_acc += get_accuracy(logits, prevLayers, data.BATCH_SIZE)

        train_running_loss /= i
        train_acc /= i

        model.eval()
        test_acc = 0.0
        test_bit_acc = 0.0
        for i, (layers, prevLayers) in enumerate(test_loader, 0):
            layers = layers.to(device)
            prevLayers = prevLayers.to(device)
            outputs = model(layers)
            test_acc += get_accuracy(outputs, prevLayers, data.BATCH_SIZE)
            test_bit_acc += get_bit_accuracy(outputs, prevLayers, data.BATCH_SIZE)

        test_acc /= i
        test_bit_acc /= i

        print('Epoch: %d | Loss: %.4f | Train Accuracy: %.2f' \
              %(epoch, train_running_loss, train_acc))
        print('Test Accuracy: %.2f | Test Bit Accuracy: %.2f'%( test_acc, test_bit_acc))
    torch.save(model, "/model.pt")

if __name__ == '__main__':
    train_loader, test_loader = data.load_data("prelim/test.csv", 2)
    train(train_loader, test_loader, model)