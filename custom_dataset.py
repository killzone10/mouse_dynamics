import torch
from torch.utils.data import Dataset
from nonlegality_analyser import * 
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        # self.path = path
        self.transform = transform
        self.data = data
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        return torch.tensor(sample, dtype=torch.float32)

    # def prepare_data(self):
    #     data = []

       
    # #     dataset = self.analyser.createTrainingDataWithLabel(self.user)
    # #     X = dataset[:, :-1]
    # #     y = dataset[:, -1]

    # #     # Separate rows based on the last column's value
    # #     x_negative = X[y == 0]
    # #     x_positive = X[y == 1]
        
    # #    ## WE CAN CHOOSE HERE WHAT SHOULD BE RETURNED

    # #     x_negative = self.scaler.fit_transform(x_negative)
    # #     x_positive = self.scaler.fit_transform(x_positive)

    #     return x_positive

