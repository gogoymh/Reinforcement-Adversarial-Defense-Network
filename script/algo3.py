import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import timeit

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
                batch_size=1, shuffle=True)


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
                batch_size=1, shuffle=False)

#####################################################################################################################
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

class Actor(nn.Module):
    def __init__(self, len_states, classes, num_actions, hidden=300):
        super(Actor, self).__init__()
        
        self.len_states = len_states
        self.classes = classes
        
        self.fc1_1 = nn.Linear(len_states, hidden)
        self.fc1_2 = nn.Linear(classes, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.fc3 = nn.Linear(hidden, num_actions)
        self.relu = nn.LeakyReLU()
        
    def forward(self, x_and_Cx):
        x, Cx = x_and_Cx
        x = x.view(-1, self.len_states)
        Cx = Cx.view(-1, self.classes)
        out = self.fc1_1(x) + self.fc1_2(Cx)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.fc3(out)
        
        return out

class Critic(nn.Module):
    def __init__(self, len_states, classes, num_actions, hidden=300):
        super(Critic, self).__init__()
        
        self.len_states = len_states
        self.classes = classes
        self.num_actions = num_actions
        
        self.fc1_1 = nn.Linear(len_states, hidden)
        self.fc1_2 = nn.Linear(classes, hidden)
        self.fc1_3 = nn.Linear(num_actions, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.fc3 = nn.Linear(hidden, 1)
        self.relu = nn.LeakyReLU()
        
    def forward(self, x_and_cx_and_action):
        x, cx, action = x_and_cx_and_action
        x = x.view(-1, self.len_states)
        cx = cx.view(-1, self.classes)
        action = action.view(-1, self.num_actions)
        out = self.fc1_1(x) + self.fc1_2(cx) + self.fc1_3(action)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

class agent_buffer:
    def __init__(self, capacity=1000000):
        self.capacity = capacity
        self.buffer = np.zeros(self.capacity, dtype=object)
        
        self.buffer_idx = 0
        self.n_entries = 0
        
        self.transition = None
    
    def go(self):
        self.transition.append(np.ones((1)))
        
    def stop(self):
        self.transition.append(np.zeros((1)))
        
    def keep(self, x, Cx, action_1, x_2, Cx_2, reward):
        self.transition = [x.detach().clone().cpu().numpy(), Cx.detach().clone().cpu().numpy(), action_1.detach().clone().cpu().numpy(), x_2.detach().clone().cpu().numpy(), Cx_2.detach().clone().cpu().numpy(), reward.detach().clone().cpu().numpy()]
        
    def add(self):
        self.buffer[self.buffer_idx] = self.transition
        
        self.buffer_idx += 1
        
        if self.buffer_idx >= self.capacity: # First in, First out
            self.buffer_idx = 0            
        
        if self.n_entries < self.capacity: # How many transitions are stored in buffer
            self.n_entries += 1
            
        self.transition = None

    def sample(self, batch_size):
        batch = []
        assert self.n_entries >= batch_size, "Buffer is not enough"
        for i in np.random.choice(self.n_entries, batch_size, replace=False):
            batch.append(self.buffer[i])
        return batch

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

#####################################################################################################################
device = torch.device("cuda:0")
Budget = 100
Confidence = 0.95
X_m = torch.zeros([10,3,32,32]).float().to(device)
alpha = 0.2
exploration = 15000
init_delta = 0.5
delta = init_delta
delta_decay = 0.95
memory_size = 20000
warmup = 10000
batch_size = 128
discount = 0.01
tau = 0.001
#checkpoint_C = torch.load("C://results//classifier.pth")
checkpoint_C = torch.load("/data1/ymh/lrcl/classifier.pth")

#####################################################################################################################
C = Classifier().to(device)
C.load_state_dict(checkpoint_C['model_state_dict'])
A = Actor(3*32*32, 10, 3*32*32).to(device)
Q = Critic(3*32*32, 10, 3*32*32).to(device)

A_t = Actor(3*32*32, 10, 3*32*32).to(device)
Q_t = Critic(3*32*32, 10, 3*32*32).to(device)
A_t.load_state_dict(A.state_dict())
Q_t.load_state_dict(Q.state_dict())

optim_C = optim.Adam(C.parameters(), lr=0.01)
optim_A = optim.Adam(A.parameters(), lr=0.0001)
optim_Q = optim.Adam(Q.parameters(), lr=0.0001)

buffer = agent_buffer(memory_size)

#####################################################################################################################
start = timeit.default_timer()
for i, (x,y) in enumerate(train_loader):
    A.eval()
    C.eval()
    x = x.to(device)
    y = y.to(device)
    for j in range(Budget):
        if (C(x).argmax() == y and C(x).max() > Confidence) or (j+1) == Budget:
            if buffer.transition is not None:
                buffer.stop()
                buffer.add()
                
            if X_m[y].sum() == 0:
                X_m[y] = x
            else:
                X_m[y] = X_m[y] + alpha * (x - X_m[y])
                
            break
            
        else:
            if buffer.transition is not None:
                buffer.go()
                buffer.add()
                
            if buffer.n_entries < exploration:
                action = torch.from_numpy(np.random.uniform(-1,1,3*32*32))
                action = action.float().to(device)
            else:
                delta = init_delta * (delta_decay ** (buffer.n_entries + 1 - exploration))
                noise = torch.from_numpy(np.random.normal(0, delta, 3*32*32))
                noise = noise.float().to(device)
                action = A([x,C(x)]).view(-1).detach().clone() + noise
                
            optim_C.zero_grad()
            action.requires_grad = True
            x_1 = x + action.view(-1,3,32,32)
            
            output = C(x_1)
            loss = F.cross_entropy(output, y)
            loss.backward(retain_graph=True)
            action_1 = action - 0.01*action.grad.data

            x_2 = x + action_1.view(-1,3,32,32)
            
            reward = -F.l1_loss(x_2, X_m[y])
            
            buffer.keep(x,C(x),action_1,x_2,C(x_2),reward)
            
            x = x_2.detach().clone()
            
    if buffer.n_entries < warmup:
        print("Buffer is warming up. [%d/%d]" % (buffer.n_entries, warmup))
    else:
        Q.train()
        A.train()
        Q_t.eval()
        A_t.eval()
            
        batch = buffer.sample(batch_size)
        batch = np.array(batch).transpose()
        
        current_state_1 = np.vstack(batch[0])
        current_state_2 = np.vstack(batch[1])
        action = np.vstack(batch[2])
        next_state_1 = np.vstack(batch[3])
        next_state_2 = np.vstack(batch[4])
        reward = np.vstack(batch[5])
        terminal = np.vstack(batch[6])
                
        current_state_1 = torch.from_numpy(current_state_1).float().to(device)
        current_state_2 = torch.from_numpy(current_state_2).float().to(device)
        action = torch.from_numpy(action).float().to(device)
        next_state_1 = torch.from_numpy(next_state_1).float().to(device)
        next_state_2 = torch.from_numpy(next_state_2).float().to(device)
        reward = torch.from_numpy(reward).float().to(device)
        terminal = torch.from_numpy(terminal).float().to(device)
        
        optim_Q.zero_grad()
        with torch.no_grad():
            critic_action_next = A_t([next_state_1, next_state_2])
            critic_Q_target_next = Q_t([next_state_1, next_state_2, critic_action_next])            
        critic_Q_target = reward + discount * critic_Q_target_next * terminal
        critic_Q_expected = Q([current_state_1, current_state_2, action])            
        critic_loss = F.mse_loss(critic_Q_expected, critic_Q_target).mean()
        critic_loss.backward()
        optim_Q.step()
            
        optim_A.zero_grad()            
        optim_Q.zero_grad()
        Q.eval()
        actor_action_expected = A([current_state_1, current_state_2])
        Q_value = Q([current_state_1, current_state_2, actor_action_expected])
        actor_loss = -Q_value.mean()
        actor_loss.backward()
        optim_A.step()
            
        soft_update(A_t, A, tau)
        soft_update(Q_t, Q, tau)
        
        if (i+1) % 1000 == 0:
            print("="*30)
            with torch.no_grad():
                A.eval()
                C.eval()
                accuracy = 0
                correct = 0
                for (x,y) in test_loader:
                    x = x.to(device)
                    y = y.to(device)
                    for k in range(Budget):
                        if C(x).max() > Confidence or (k+1) == Budget:
                            pred = C(x).argmax(1, keepdim=True)
                            break
                        else:
                            action = A([x,C(x)]).view(-1)
                
                            x_1 = x + action.view(-1,3,32,32)
                
                            x = x_1.clone()
                    
                    correct += pred.eq(y.long().to(device).view_as(pred)).sum().item()  
        
            accuracy = correct / len(test_loader.dataset)
            print("[Iteration:%d][Test Accuracy:%f] [Delta:%f]" % ((i+1), accuracy, delta))

finish = timeit.default_timer()
print("[Time:%f]" % (finish-start))


















