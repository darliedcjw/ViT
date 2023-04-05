import numpy as np

from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor, Compose
from torchvision.datasets.mnist import MNIST

from vit import ViT

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

    model = ViT(image_res=(1, 28, 28), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10).to(device)
    epochs = 5
    lr = 0.005

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()

    for epoch in trange(epochs, desc='Training'):
        
        train_loss = 0.0
        train_acc = 0.0

        # Training
        model.train()

        for batch in tqdm(train_loader, desc='Epoch {} in training'.format(epoch+1)):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)

            train_loss += loss.detach().cpu().item()

            train_acc += torch.sum(torch.argmax(y_pred, dim=1) == y).detach().cpu().item() / len(y)

            optimizer.zero_grad()
            
            loss.backward()

            optimizer.step()

        print('Epoch {}: \nTrain Loss - {} \nAccuracy - {}'.format(epoch + 1, train_loss / len(train_loader), train_acc / len(train_loader)))

        # Eval
        model.eval()

        with torch.no_grad():
            val_loss = 0.0
            val_accuracy = 0.0
            for batch in tqdm(val_loader, desc='Validating'):
                x, y = batch
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss = criterion (y_pred, y)

                val_loss += loss.detach().cpu().item()

                val_accuracy += torch.sum(torch.argmax(y_pred, dim=1) == y).detach().cpu().item() / len(y)
            
        print('Epoch {}: \nTrain Loss - {} \nAccuracy - {}'.format(epoch + 1, val_loss / len(val_loader), val_accuracy / len(val_loader)))

if __name__ == '__main__':
    main()

    