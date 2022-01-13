import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from scipy import stats

class Actor(nn.Module):
    def __init__(self, len_states, num_actions, hidden=1000, middle=1000):
        super(Actor, self).__init__()
        self.fc1_1 = nn.Linear(1, hidden)
        self.fc1_2 = nn.Linear(len_states, hidden) # state index
        self.fc2 = nn.Linear(hidden, middle)
        #self.fc3 = nn.Linear(middle, middle)
        self.fc4 = nn.Linear(middle, hidden)
        self.fc5 = nn.Linear(hidden, num_actions)
        self.relu = nn.LeakyReLU()
        
    def forward(self, index_and_state):
        index, state = index_and_state
        index = index.view(-1, 1)
        state = state.view(-1, 3*32*32)
        out = self.fc1_1(index) + self.fc1_2(state)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        #out = self.fc3(out)
        #out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)

        return out

class Critic(nn.Module):
    def __init__(self, len_states, num_actions, hidden=1000, middle=1000):
        super(Critic, self).__init__()
        self.fc1_1 = nn.Linear(1, hidden)
        self.fc1_2 = nn.Linear(len_states, hidden)
        self.fc1_3 = nn.Linear(num_actions, hidden)
        self.fc2 = nn.Linear(hidden, middle)
        #self.fc3 = nn.Linear(middle, middle)
        self.fc4 = nn.Linear(middle, hidden)
        self.fc5 = nn.Linear(hidden, 1)
        self.relu = nn.LeakyReLU()
        
    def forward(self, state_and_action):
        index, state, action = state_and_action
        index = index.view(-1, 1)
        state = state.view(-1, 3*32*32)
        action = action.view(-1, 84)
        out = self.fc1_1(index) + self.fc1_2(state) + self.fc1_3(action)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        #out = self.fc3(out)
        #out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        return out
    
class Agent:
    def __init__(self):
        return
    
    def searching_action(self, index, state):
        return
    
    def deterministic_action(self, index, state):
        return
    
    def add_agent_memory(self, current_state, action, next_state, reward, terminal, state_idx):
        return
        
    def update_actor_critic(self):
        return

































