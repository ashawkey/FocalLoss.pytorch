import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.optim import lr_scheduler
import numpy as np
from easydict import EasyDict

from focalloss import FocalLoss
from sampler import ImbalancedDatasetSampler

from crucible.trainer import Trainer
from crucible.metrics import *
from crucible.io import logger
from crucible.utils import fix_random_seed

class fcdr(nn.Module):
    def __init__(self, Fin, Fout, dp=0.5):
        super(fcdr, self).__init__()
        self.fc = nn.Linear(Fin, Fout)
        self.dp = nn.Dropout(dp)
        self.ac = nn.ReLU(True)
    def forward(self, x):
        x = self.fc(x)
        x = self.dp(x)
        x = self.ac(x)
        return x

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fc0 = fcdr(10, 256)
        self.fc1 = fcdr(256, 512) 
        self.fc2 = nn.Linear(512, 2)
    
    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x) # [B, 2]
        return x

def gendata(N, p=[0.9, 0.1]):
    Y = np.random.choice(2, size=(N, 1), p=p).astype(np.int64)
    X = (np.random.rand(N, 10) + Y * 0.2).astype(np.float32)
    Y = torch.LongTensor(Y)
    X = torch.FloatTensor(X)
    return X, Y

class BiasedDataset(data.Dataset):
    def __init__(self, N, p=[0.9, 0.1]):
        super().__init__()
        self.N = N
        self.p = p
        Y = np.random.choice(2, size=(N, 1), p=p).astype(np.int64)
        X = (np.random.rand(N, 10) + Y * 0.2).astype(np.float32)
        self.Y = torch.LongTensor(Y).squeeze()
        self.X = torch.FloatTensor(X)

    def __getitem__(self, index):
        data = {
            "input": self.X[index],
            "truth": self.Y[index],
            }

        return data

    def __len__(self):
        return self.N

if __name__ == "__main__":
    conf = EasyDict()
    conf.workspace = 'workspace/imbalanced_sampler_cross_entropy'
    conf.device = 'cuda'
    conf.max_epochs = 100
    
    log = logger(conf.workspace)
    model = FCN()
    #loss_function = FocalLoss()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    metrics = [ClassificationMeter(2),]
    train_dataset = BiasedDataset(1000, [0.9, 0.1])
    test_dataset = BiasedDataset(1000, [0.5, 0.5])
    loaders = {
            "train": data.DataLoader(train_dataset, sampler=ImbalancedDatasetSampler(train_dataset)),
            "test": data.DataLoader(test_dataset),
            }


    trainer = Trainer(conf, model, optimizer, scheduler, loss_function, loaders, log, metrics)
    trainer.train()
    trainer.evaluate()

