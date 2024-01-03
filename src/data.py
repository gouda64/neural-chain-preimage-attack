import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

BATCH_SIZE = 64
INITIAL_STATE = "67452301efcdab8998badcfe10325476c3d2e1f0"

## transformations
transform = transforms.Compose(
    [transforms.ToTensor()])

class HashDataset(Dataset):
    def __init__(self, csv_file, layer_num = 1, transform=None):
        # layer 1 is from 1 -> 0
        self.hash_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.layer_num = layer_num

    def __len__(self):
        return len(self.hash_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input = self.hash_frame.iloc[idx, self.layer_num]
        output = INITIAL_STATE if (self.layer_num == 1) else self.hash_frame.iloc[idx, self.layer_num - 1]
        sample = {'layer': input, 'prevLayer': output}

        if self.transform:
            sample = self.transform(sample)

        return sample

hash_dataset = HashDataset(csv_file="data.csv", layer_num=2)

train_dataset=hash_dataset.sample(frac=0.8,random_state=200)
test_dataset=hash_dataset.drop(train_dataset.index).sample(frac=1.0)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

print("running")