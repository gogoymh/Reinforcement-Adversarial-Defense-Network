# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 12:30:36 2020

@author: Minhyeong
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

np.random.seed(1)
class_number = 100
test_number = 50

red_mean = [0.5,0.5]
red_cov = [[0.3,-0.01],[-0.01,0.3]]
red = np.ones((class_number,2))
red[:,0:2] = np.transpose(np.random.multivariate_normal(red_mean, red_cov, class_number).T + np.random.uniform(-0.3,0.3, class_number))
red_t = np.ones((test_number,2))
red_t[:,0:2] = np.transpose(np.random.multivariate_normal(red_mean, red_cov, test_number).T + np.random.uniform(-0.3,0.3, test_number))

blue_mean = [0.5,-0.5]
blue_cov = [[0.3,0],[0,0.3]]
blue = np.ones((class_number,2))
blue[:,0:2] = np.transpose(np.random.multivariate_normal(blue_mean, blue_cov, class_number).T + np.random.uniform(-0.3,0.3, class_number))
blue_t = np.ones((test_number,2))
blue_t[:,0:2] = np.transpose(np.random.multivariate_normal(blue_mean, blue_cov, test_number).T + np.random.uniform(-0.3,0.3, test_number))

green_mean = [-0.5,-0.5]
green_cov = [[0.3,0.015],[0.015,0.3]]
green = np.ones((class_number,2))
green[:,0:2] = np.transpose(np.random.multivariate_normal(green_mean, green_cov, class_number).T + np.random.uniform(-0.3,0.3, class_number))
green_t = np.ones((test_number,2))
green_t[:,0:2] = np.transpose(np.random.multivariate_normal(green_mean, green_cov, test_number).T + np.random.uniform(-0.3,0.3, test_number))

yellow_mean = [-0.5,0.5]
yellow_cov = [[0.3,0],[0,0.3]]
yellow = np.ones((class_number,2))
yellow[:,0:2] = np.transpose(np.random.multivariate_normal(yellow_mean, yellow_cov, class_number).T + np.random.uniform(-0.3,0.3, class_number))
yellow_t = np.ones((test_number,2))
yellow_t[:,0:2] = np.transpose(np.random.multivariate_normal(yellow_mean, yellow_cov, test_number).T + np.random.uniform(-0.3,0.3, test_number))

colors = ["red", "blue", "green", "yellow"]

plt.scatter(red[:,0], red[:,1], color=colors[0], marker="o", s=5)
plt.scatter(blue[:,0], blue[:,1], color=colors[1], marker="o", s=5)
plt.scatter(green[:,0], green[:,1], color=colors[2], marker="o", s=5)
plt.scatter(yellow[:,0], yellow[:,1], color=colors[3], marker="o", s=5)

plt.axvline(x=0, ymin=-2, ymax=2, color='black', linestyle='--', linewidth=0.5)
plt.axhline(y=0, xmin=-2, xmax=2, color='black', linestyle='--', linewidth=0.5)
plt.axis([-2,2,-2,2])
plt.show()
plt.close()
#plt.savefig('C://유민형//개인 연구//Reinforcement Adversarial Defense Network//result//train.png')

plt.scatter(red_t[:,0], red_t[:,1], color=colors[0], marker="o", s=5)
plt.scatter(blue_t[:,0], blue_t[:,1], color=colors[1], marker="o", s=5)
plt.scatter(green_t[:,0], green_t[:,1], color=colors[2], marker="o", s=5)
plt.scatter(yellow_t[:,0], yellow_t[:,1], color=colors[3], marker="o", s=5)

plt.axvline(x=0, ymin=-2, ymax=2, color='black', linestyle='--', linewidth=0.5)
plt.axhline(y=0, xmin=-2, xmax=2, color='black', linestyle='--', linewidth=0.5)
plt.axis([-2,2,-2,2])
plt.show()
plt.close()
#plt.savefig('C://유민형//개인 연구//Reinforcement Adversarial Defense Network//result//test.png')


data = np.concatenate((red,blue,green,yellow))
label = np.zeros((4*class_number))
label[class_number:2*class_number] = 1
label[2*class_number:3*class_number] = 2
label[3*class_number:4*class_number] = 3

data_t = np.concatenate((red_t,blue_t,green_t,yellow_t))
label_t = np.zeros((4*test_number))
label_t[test_number:2*test_number] = 1
label_t[2*test_number:3*test_number] = 2
label_t[3*test_number:4*test_number] = 3

data = torch.from_numpy(data).float()
label = torch.from_numpy(label).long()
data_t = torch.from_numpy(data_t).float()
label_t = torch.from_numpy(label_t).long()

dataset = TensorDataset(data, label)
dataset_t = TensorDataset(data_t, label_t)

train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(dataset_t, batch_size=50, shuffle=False)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        
        mid = 100
        self.layer1 = nn.Linear(2,mid)
        self.norm1 = nn.BatchNorm1d(mid)
        self.layer2 = nn.Linear(mid,mid)
        self.norm2 = nn.BatchNorm1d(mid)
        self.layer3 = nn.Linear(mid,mid)
        self.norm3 = nn.BatchNorm1d(mid)
        self.layer4 = nn.Linear(mid, mid)
        self.norm4 = nn.BatchNorm1d(mid)
        self.layer5 = nn.Linear(mid,mid)
        self.norm5 = nn.BatchNorm1d(mid)
        self.layer6 = nn.Linear(mid,4)
        
        self.activation = nn.LeakyReLU()
        self.output = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        out = self.layer1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.layer2(out)
        out = self.norm2(out)
        out = self.activation(out)
        out = self.layer3(out)
        out = self.norm3(out)
        out = self.activation(out)
        out = self.layer4(out)
        out = self.norm4(out)
        out = self.activation(out)
        out = self.layer5(out)
        out = self.norm5(out)
        out = self.activation(out)
        out = self.layer6(out)
        out = self.output(out)
        
        return out

criterion = nn.CrossEntropyLoss()

model = Classifier()
device = torch.device("cuda:0")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("="*100)
losses = torch.zeros((100))
for epoch in range(100):
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

xx = [i/100 for i in range(-199,200)]
db = torch.zeros([399*399,2])
for i in range(399):
    db[399*i:399*(i+1),0] = torch.Tensor(xx)
    db[399*i:399*(i+1),1] = torch.Tensor(xx)[i]

db = TensorDataset(db)

db_loader = DataLoader(db, batch_size=64, shuffle=True)

with torch.no_grad():
    model.eval()
    for x in db_loader:
        x = x[0]
        output = model(x.float().to(device))
        confidence = output.max(1, keepdim=True)[0]
        confidence = confidence.view(-1).cpu().numpy()
        pred = output.argmax(1, keepdim=True)
        for_c = []
        for i in pred:
            for_c.append(colors[i])
        for_s = 0.01 * np.ones((x.shape[0]))
        for_s[confidence > 0.9] = 0.1
        for_p = x.cpu().numpy()
        plt.scatter(for_p[:,0], for_p[:,1], s = for_s, c = for_c)
    
    for x, y in train_loader:
        output = model(x.float().to(device))
        confidence = output.max(1, keepdim=True)[0]
        pred = output.argmax(1, keepdim=True)
        for_c = []
        for i in y:
            for_c.append(colors[i])        
        for_s = ((confidence.cpu().numpy()-0.5)/0.5) * 100
        for_p = x.cpu().numpy()
        plt.scatter(for_p[:,0], for_p[:,1], s = for_s, c = for_c)
    
    
    
    plt.axvline(x=0, ymin=-2, ymax=2, color='black', linestyle='--', linewidth=0.5)
    plt.axhline(y=0, xmin=-2, xmax=2, color='black', linestyle='--', linewidth=0.5)
    plt.axis([-2,2,-2,2])
    plt.show()
    plt.close()
    

with torch.no_grad():
    model.eval()
    for x in db_loader:
        x = x[0]
        output = model(x.float().to(device))
        confidence = output.max(1, keepdim=True)[0]
        confidence = confidence.view(-1).cpu().numpy()
        pred = output.argmax(1, keepdim=True)
        for_c = []
        for i in pred:
            for_c.append(colors[i])
        for_s = 0.01 * np.ones((x.shape[0]))
        for_s[confidence > 0.9] = 0.1
        for_p = x.cpu().numpy()
        plt.scatter(for_p[:,0], for_p[:,1], s = for_s, c = for_c)
        
    for x, y in test_loader:
        output = model(x.float().to(device))
        confidence = output.max(1, keepdim=True)[0]
        pred = output.argmax(1, keepdim=True)
        for_c = []
        for i in y:
            for_c.append(colors[i])        
        for_s = ((confidence.cpu().numpy()-0.5)/0.5) * 100
        for_p = x.cpu().numpy()
        plt.scatter(for_p[:,0], for_p[:,1], s = for_s, c = for_c)
    
    
        
    plt.axvline(x=0, ymin=-2, ymax=2, color='black', linestyle='--', linewidth=0.5)
    plt.axhline(y=0, xmin=-2, xmax=2, color='black', linestyle='--', linewidth=0.5)
    plt.axis([-2,2,-2,2])
    plt.show()
    plt.close()


