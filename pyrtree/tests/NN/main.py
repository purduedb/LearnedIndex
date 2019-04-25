import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.autograd import Variable
from dataloader import RtreeMappingDataset
from model import Net
import argparse

parser = argparse.ArgumentParser(description='Neural Net Training Script')
parser.add_argument('-t', '--trainfile', type=str,
                    default='../NNdisjointdataset.csv', help='Training Set File')
parser.add_argument('-v', '--valfile', type=str,
                    default='../NNtestdisjointdataset.csv', help='Validation Set File')

args = parser.parse_args()

rdataset = RtreeMappingDataset(csv_file=args.trainfile, epsilon=0.015,
                transform=None)
rdataset_test = RtreeMappingDataset(csv_file=args.valfile, epsilon=0.015,
                transform=None)

kwargs = {}
train_loader = torch.utils.data.DataLoader(
    rdataset, batch_size=256, shuffle=True, **kwargs
    )

test_loader = torch.utils.data.DataLoader(
    rdataset_test, batch_size=256, shuffle=True, **kwargs
    )


model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
#        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
            output = model(data)
#            test_loss += criterion(output, target)
            test_loss += F.nll_loss(output, target, size_average=False).data.item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
#            print(pred.eq(target.data.view_as(pred)).squeeze())

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return (100. * correct / len(test_loader.dataset))

def test_plot(best_model):
    best_model.eval()
    test_loss = 0
    correct = 0
    pts = []
    lbls = []
    Y = []
    for data, target in test_loader:
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
            output = best_model(data)
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            pts.extend(data.data.numpy().tolist())
            lbls.extend(pred.eq(target.data.view_as(pred)).squeeze().numpy())
            Y.extend(pred.squeeze().numpy())

    test_loss /= len(train_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    pts = np.asarray(pts)

    # Plot Binary Green/Red
    plt.scatter(pts[:, 0], pts[:, 1], alpha=0.3, s=10, c=lbls, cmap='RdYlGn')
    # Plot HSV Colormap
    #plt.scatter(pts[:, 0], pts[:, 1], alpha=0.3, s=10, c=Y, vmin=0, vmax=99, cmap='hsv')
    plt.savefig("Plot.png")

    return (100. * correct / len(test_loader.dataset))


best_acc = 0.0
for epoch in range(1, 80 + 1):
#    if epoch%15 == 0:
#        for param_group in optimizer.param_groups:
#            param_group['lr'] *= 0.5

    train(epoch)
    acc = test().item()
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'best_model.pth')

print("Best Model Acc: ", best_acc)
best_model = Net()
best_model.load_state_dict(torch.load('best_model.pth'))
test_plot(best_model)
