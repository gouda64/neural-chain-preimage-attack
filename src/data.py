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
    [ transforms.ToTensor()])

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

        input = self.hash_frame.iloc[idx, 1 if (self.layer_num == 0) else self.layer_num]
        output = INITIAL_STATE if (self.layer_num == 1) else self.hash_frame.iloc[idx, 0 if (self.layer_num == 0) else self.layer_num - 1]
        # print(input)
        # print(output)
        sample = {'layer': torch.tensor([float(char) for char in format(int(input, 16), '0>160b')]),
                  'prevLayer': torch.tensor([float(char) for char in format(int(output, 16), '0>160b')])}

        if self.transform:
            sample = self.transform(sample)

        return sample['layer'], sample['prevLayer']


def load_data(csv_file, layer):
    hash_dataset = HashDataset(csv_file=csv_file, layer_num=layer)
    print("dataset loaded")

    train_dataset, test_dataset = torch.utils.data.random_split(hash_dataset, [0.8, 0.2])
    print("dataset split")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    print("loaders loaded")

    return train_loader, test_loader

if __name__ == '__main__':
    train_loader, test_loader = load_data("prelim/test.csv", 2)
    print(list(train_loader)[0])
    print(list(test_loader)[0])
    #
    # input = "07893e555712ce8a14ae47e0a8ed3174d4053666"
    # print(torch.tensor([int(char) for char in format(int(input, 16), '0>432b')]))