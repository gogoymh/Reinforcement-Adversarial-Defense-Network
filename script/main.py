import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from scipy import stats
import argparse

from environment import Environment

#####################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
parser.add_argument("--budget", type=int, default=100, help="Budget")
parser.add_argument("--threshold", type=float, default=0.8, help="Threshold")

'''
parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
parser.add_argument("--memory_size", type=int, default=2000, help="Memory size")
parser.add_argument("--warmup", type=int, default=1000, help="Warm up episodes")
parser.add_argument("--exploration", type=int, default=100, help="Total episodes")
parser.add_argument("--exploitation", type=int, default=300, help="Total episodes")

parser.add_argument("--init_delta", type=float, default=0.5, help="Initial delta")
parser.add_argument("--delta_decay", type=float, default=0.95, help="Delta decay")
parser.add_argument("--discount", type=float, default=1, help="Discount factor for Q-value function")
parser.add_argument("--tau", type=float, default=0.01, help="Tau for soft update")

parser.add_argument("--alpha", type=float, default=0.05, help="Search parameter")
'''

opt = parser.parse_args()
print("="*100)
print(opt)
print("="*100)

#####################################################################################################################
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
                batch_size=1, shuffle=True, pin_memory=True)


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
                batch_size=100, shuffle=False, pin_memory=True)

#####################################################################################################################
env = Environment(train_loader, opt)




