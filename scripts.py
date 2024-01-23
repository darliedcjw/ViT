import os
from datetime import datetime
import numpy as np

from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor, Compose
from torchvision.datasets.mnist import MNIST
from torchvision.datasets import ImageFolder

from vit import ViT

np.random.seed(0)
torch.manual_seed(0)

def main():
    transform = Compose([
        ToTensor(),
    ])

    train = ImageFolder('datasets/numbers/train', transform=transform)
    val = ImageFolder('datasets/numbers/val', transform=transform)

    train_loader = DataLoader(train, shuffle=True, batch_size=1)
    val_loader = DataLoader(val, shuffle=False, batch_size=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using: {}'.format(device))

    model = ViT(image_res=(3, 28, 28), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=5).to(device)
    epochs = 200
    lr = 0.0005

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()

    best_val_loss = .0
    best_val_acc = .0

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

        print('Epoch {}: \nTrain Loss - {} \nTrain Accuracy - {}'.format(epoch + 1, train_loss / len(train_loader), train_acc / len(train_loader)))

        # Eval
        model.eval()

        with torch.no_grad():
            val_loss = 0.0
            val_acc = 0.0
            for batch in tqdm(val_loader, desc='Validating'):
                x, y = batch
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss = criterion (y_pred, y)

                val_loss += loss.detach().cpu().item()

                val_acc += torch.sum(torch.argmax(y_pred, dim=1) == y).detach().cpu().item() / len(y)

        print('Epoch {}: \Validation Loss - {} \nValidation Accuracy - {}'.format(epoch + 1, val_loss / len(val_loader), val_acc / len(val_loader)))

        # Save
        if not os.path.exists('weights'):
            os.makedirs('weights')
        
        if best_val_loss == 0.0 and best_val_acc == 0.0:
            folder = datetime.strftime(datetime.now(), format='%Y%m%d_%H%M%S') 
            os.makedirs('weights/{}'.format(folder))

            best_val_loss = val_loss / len(val_loader)
            best_val_acc = val_acc / len(val_loader)
            
            print("SAVING BEST LOSS: {:.2f}".format(best_val_loss))
            torch.save(model.state_dict(), 'weights/{}/best_val_loss_{:.2f}.pth'.format(folder, best_val_loss))
            
            print("SAVING BEST ACC: {:.2f}".format(best_val_acc))
            torch.save(model.state_dict(), 'weights/{}/best_val_acc_{:.2f}.pth'.format(folder, best_val_acc))

        
        if (val_loss / len(val_loader)) < best_val_loss:
            best_val_loss = val_loss / len(val_loader)
            print("SAVING BEST LOSS: {:.2f}".format(best_val_loss))
            torch.save(model.state_dict(), 'weights/{}/best_val_loss_{:.2f}.pth'.format(folder, best_val_loss))

        if (val_acc / len(val_loader)) > best_val_acc:
            best_val_acc = val_acc / len(val_loader)
            print("SAVING BEST ACC: {:.2f}".format(best_val_acc))
            torch.save(model.state_dict(), 'weights/{}/best_val_acc_{:.2f}.pth'.format(folder, best_val_acc))

if __name__ == '__main__':
    main()

    