import os
import torch
import torchvision
import tarfile
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib
import matplotlib.pyplot as plt

import torch.cuda
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from DeepNeuralNetworkClasses import *
if __name__ == '__main__':
    dataset = MNIST(root='data/', download=True, transform=ToTensor())

    train_dataset, test_dataset = random_split(dataset, [int(0.9 * len(dataset)), int(0.1 * len(dataset))])

    batch_size = 350
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=4, pin_memory=True)

    in_size = 28*28
    hidden_size = 28*28*5
    out_size = 10

    epochs = 3
    lr =0.1

    device = get_default_device()
    print(device)
    train_dataloader = DeviceDataLoader(train_dataloader, device)

    model = MNISTModelv1(in_size, hidden_size, out_size)
    model1 = MNISTModelv0(in_size, hidden_size, out_size)
    model1 = to_device(model1, device)
    model = to_device(model, device)
    torch.cuda.empty_cache()
    optimizer = torch.optim.SGD
    # evaluate(train_dataloader, model)
    torch.cuda.empty_cache()
    fit(epochs, lr, train_dataloader, test_dataloader, optimizer, model)
    print("second model")
    fit(epochs, lr, train_dataloader, test_dataloader, optimizer, model1)
