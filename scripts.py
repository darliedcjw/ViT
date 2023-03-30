import numpy as np

from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor, Compose
from torchvision.datasets.mnist import MNIST

np.random.seed(0)
torch.manual_seed(0)

def main():
    transform = Compose([
        ToTensor(),
    ])

    train = MNIST(root='./datasets', train=True, download=True, transform=transform)
    val = MNIST(root='./datasets', train=False, download=True, transform=transform)

    train_loader = DataLoader(train, shuffle=True, batch_size=128)
    val_loader = DataLoader(val, shuffle=False, batch_size=128)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using: {}'.format(device))
    