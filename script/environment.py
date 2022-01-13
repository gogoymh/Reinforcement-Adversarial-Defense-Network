import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from scipy import stats

class Filter(nn.Module):
    def __init__(self):
        super(Filter, self).__init__()
        
        self.filter = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.activation = nn.Tanh() # to scale for classifier
        
    def forward(self, x):
        
        out = self.filter(x)
        out = self.activation(out)
        
        return out

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        
        self.layer1_1 = nn.Linear(1,768)
        self.layer1_2 = nn.Conv2d(3, 3, kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Linear(768, 100)
        self.layer3 = nn.Linear(100, 10)
        
        self.activation = nn.LeakyReLU()
        self.output = nn.Softmax(dim=1)
        
    def forward(self, index_and_state):
        index = index_and_state[0].view(-1, 1)
        state = index_and_state[1].view(-1, 3, 32, 32)
        out_1 = self.layer1_1(index)
        out_2 = self.layer1_2(state)
        out_2 = out_2.view(-1, 768)
        out = out_1 + out_2
        out = self.activation(out)
        out = self.layer2(out)
        out = self.activation(out)
        out = self.layer3(out)
        out = self.output(out)
        
        return out

class Environment:
    def __init__(self, train_loader, options):
        self.train_loader = train_loader
        self.device = options.device
        self.budget = options.budget
        self.threshold = options.threshold
        
        self.model = Filter().to(self.device)
        self.model_optim = optim.Adam(self.model.parameters(), lr=0.1)
        self.model_list = list(self.model.modules())
        
        self.classifier = Classifier().to(self.device)
        self.classifier_optim = optim.Adam(self.classifier.parameters(), lr=0.1)
        
        #self.classifier_memory = 
        
    def reset(self):
        self.current_index = 0
        self.current_state, self.ground_truth = self.train_loader.__iter__().next()
        self.current_state = self.current_state.to(self.device)
        self.ground_truth = self.ground_truth.to(self.device)
        
        while self.check(False):
            self.current_state, self.ground_truth = self.train_loader.__iter__().next()
            self.current_state = self.current_state.to(self.device)
            self.ground_truth = self.ground_truth.to(self.device)
        
        return [torch.Tensor([self.current_index/self.budget]).to(self.device), self.current_state]
        
    def step(self, action):
        action_1 = torch.from_numpy(action[:,:81]).view(3,3,3,3)
        action_2 = torch.from_numpy(action[:,81:]).view(-1)
                
        self.model_list[1].weight.data = action_1.clone().float().to(self.device)
        self.model_list[1].bias.data = action_2.clone().float().to(self.device)
        
        self.model_optim.zero_grad()
        next_state = self.model(self.current_state)
        output = self.classifier([torch.Tensor([(self.current_index + 1)/self.budget]).to(self.device), next_state])
        loss = F.cross_entropy(output, self.ground_truth)
        loss.backward(retain_graph=True)
        self.model_optim.step()
                
        action_1 = self.model_list[1].weight.data.clone().view(-1)
        action_2 = self.model_list[1].bias.data.clone().view(-1)
        revised_action = torch.cat((action_1, action_2))
                
        next_state = self.model(self.current_state)
        self.current_index += 1
        self.current_state = next_state
        
        reward, terminal = self.check()
        
        return revised_action, [torch.Tensor([self.current_index/self.budget]).to(self.device), self.current_state], reward, terminal
    
    def check(self, get_reward=True):
        output = self.classifier([torch.Tensor([self.current_index/self.budget]).to(self.device), self.current_state])
        done = output.max(1, keepdim=True)[0] >= self.threshold
        
        if get_reward:
            reward = - F.cross_entropy(output, self.ground_truth)
            if done:
                terminal = np.zeros((1))
            else:
                if self.current_index == self.budget:
                    terminal = np.zeros((1))
                else:
                    terminal = np.ones((1))
        
            return reward, terminal
        
        else:
            return done
        
    def add_classifier_memory(self,x,y):
        return
    
    def update_classifier(self):
        return
        

































