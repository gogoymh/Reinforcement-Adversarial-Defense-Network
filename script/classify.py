import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

train_loader = DataLoader(
                datasets.CIFAR10(
                        "../data/CIFAR10",
                        train=True,
                        download=True,
                        transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                                ),
                        ),
                batch_size=128, shuffle=True)


test_loader = DataLoader(
                datasets.CIFAR10(
                        '../data/CIFAR10',
                        train=False,
                        download=True,
                        transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                                ),
                        ),
                batch_size=100, shuffle=False)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        
        self.layer1 = nn.Conv2d(3, 3, kernel_size=4, stride=2, padding=1)
        self.norm1 = nn.BatchNorm2d(3)
        self.layer2 = nn.Linear(768, 100)
        self.norm2 = nn.BatchNorm1d(100)
        self.layer3 = nn.Linear(100, 10)
        
        self.activation = nn.LeakyReLU()
        self.output = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        out = self.layer1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = out.view(-1, 768)
        out = self.layer2(out)
        out = self.norm2(out)
        out = self.activation(out)
        out = self.layer3(out)
        out = self.output(out)
        
        return out
    
    

criterion = nn.CrossEntropyLoss()

model = Classifier()
device = torch.device("cuda:0")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("="*100)
losses = torch.zeros((300))
for epoch in range(300):
    accuracy = 0
    correct = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(x.to(device))
        pred = output.argmax(1, keepdim=True)
        correct += pred.eq(y.long().to(device).view_as(pred)).sum().item()  
        loss = criterion(output, y.to(device))
        loss.backward()
        optimizer.step()
        losses[epoch] += loss.item()
    losses[epoch] /= len(train_loader)
    accuracy = correct / len(train_loader.dataset)
    print("[Epoch:%d] [Loss:%f] [Train Accuracy:%f]" % (epoch+1, losses[epoch], accuracy), end=" ")
    
    accuracy = 0
    with torch.no_grad():
        model.eval()
        correct = 0
        for x, y in test_loader:
            output = model(x.float().to(device))
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(y.long().to(device).view_as(pred)).sum().item()                
        accuracy = correct / len(test_loader.dataset)
        print("[Test Accuracy:%f]" % accuracy)
        model.train()

torch.save({'model_state_dict': model.state_dict()}, "C://results//classifier.pth")

