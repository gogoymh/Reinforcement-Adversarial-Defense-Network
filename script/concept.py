import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from scipy import stats

#####################################################################################################################
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
        
        self.layer1 = nn.Conv2d(3, 3, kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Linear(768, 100)
        self.layer3 = nn.Linear(100, 10)
        
        self.activation = nn.LeakyReLU()
        self.output = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        out = self.layer1(x)
        out = self.activation(out)
        out = out.view(-1, 768)
        out = self.layer2(out)
        out = self.activation(out)
        out = self.layer3(out)
        out = self.output(out)
        
        return out

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
        
        #weight = out[:,:81].view(-1,3,3,3,3)
        #bias = out[:,81:].view(-1,3)

        return out #weight, bias

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
'''
#####################################################################################################################
device = torch.device("cuda:0")
criterion = nn.CrossEntropyLoss()

model = Filter().to(device)
optimizer_model = optim.Adam(model.parameters(), lr=0.01)
model_list = list(model.modules())

classifier = Classifier().to(device)
optimizer_classifier = optim.Adam(classifier.parameters(), lr=0.01)

actor = Actor(3*32*32, 84).to(device)
actor_target = Actor(3*32*32, 84).to(device)
optimizer_actor = optim.Adam(actor.parameters(), lr=0.01)

critic = Critic(3*32*32, 84).to(device)
critic_target = Critic(3*32*32, 84).to(device)
optimizer_critic = optim.Adam(critic.parameters(), lr=0.01)

actor_target.load_state_dict(actor.state_dict())
critic_target.load_state_dict(critic.state_dict())

#####################################################################################################################
class classifier_buffer:
    def __init__(self, capacity=1000000):
        self.capacity = capacity
        self.buffer_x = torch.zeros((capacity,3,32,32))
        self.buffer_y = torch.zeros((capacity))
        
        self.buffer_idx = 0
        self.n_entries = 0
    
    def add(self, X, Y):
        for i in range(X.shape[0]):
            self.single_add(X[i], Y[i])
    
    def single_add(self, x, y):
        self.buffer_x[self.buffer_idx] = x.detach().clone().cpu()
        self.buffer_y[self.buffer_idx] = y.detach().clone().cpu()
        
        self.buffer_idx += 1
        
        if self.buffer_idx >= self.capacity: # First in, First out
            self.buffer_idx = 0            
        
        if self.n_entries < self.capacity: # How many transitions are stored in buffer
            self.n_entries += 1
        
    def sample(self, batch_size):
        assert self.n_entries >= batch_size, "Buffer is not enough"
        index = np.random.choice(self.n_entries, batch_size, replace=False)
        
        return self.buffer_x[index], self.buffer_y[index]

class agent_buffer:
    def __init__(self, capacity=1000000):
        self.capacity = capacity
        self.buffer = np.zeros(self.capacity, dtype=object)
        self.index = np.zeros(self.capacity, dtype=object)
        
        self.buffer_idx = 0
        self.n_entries = 0
    
    def add(self, current_state, action, next_state, reward, terminal, index):
        transition = [current_state.detach().clone().cpu().numpy(), action.detach().clone().cpu().numpy(), next_state.detach().clone().cpu().numpy(), reward.detach().clone().cpu().numpy(), terminal]
        self.buffer[self.buffer_idx] = transition
        self.index[self.buffer_idx] = index
        
        self.buffer_idx += 1
        
        if self.buffer_idx >= self.capacity: # First in, First out
            self.buffer_idx = 0            
        
        if self.n_entries < self.capacity: # How many transitions are stored in buffer
            self.n_entries += 1

    def sample(self, batch_size):
        batch = []
        index = []
        assert self.n_entries >= batch_size, "Buffer is not enough"
        for i in np.random.choice(self.n_entries, batch_size, replace=False):
            batch.append(self.buffer[i])
            index.append(self.index[i])
        return batch, index

#####################################################################################################################
def sample_from_truncated_normal_distribution(lower, upper, mu, sigma, size=1):
    return stats.truncnorm.rvs((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma, size=size)

def truncated_random_action(mean, var):
    revised_action = np.zeros((mean.shape[0]))
    for i in range(mean.shape[0]):
        revised_action[i] = sample_from_truncated_normal_distribution(lower=-1, upper=1, mu=mean[i], sigma=var)
    return revised_action

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

#####################################################################################################################
# Hyper parameter
batch_size = 128
threshold = 0.8
budget = 100
memory_size = 2000
warmup = 1000
exploration = 5
exploitation = 300
init_delta = 0.5
delta = init_delta
delta_decay = 0.95
moving_average = None
moving_alpha = 0.5
discount = 1
tau = 0.01

#####################################################################################################################
classifier_memory = classifier_buffer(memory_size)
agent_memory = agent_buffer(memory_size)

#####################################################################################################################
for epoch in range(exploration + exploitation):
    print("="*100)
    print('[Epoch: %d]' % (epoch+1), end=" ")
    print('[delta: %f]' % delta, end=" ")
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        state_idx = 0
        
        output = classifier(x)
        done = output.max(1, keepdim=True)[0]
        
        while (done < threshold).sum().item():
            classifier_memory.add(x, y)
            
            x = x[(done < threshold).reshape(-1)]
            y = y[(done < threshold).reshape(-1)]
            state_idx_tensor = torch.Tensor([state_idx/budget]).to(device)
            
            for i in range(x.shape[0]):
                current_state = x[i].view(1,3,32,32)
                ground_truth = y[i].view(1)
                
                action = actor([state_idx_tensor, current_state])
                action = action.view(-1)
                action = action.clone().detach().cpu().numpy()
                
                if (epoch + 1) > exploration: # exploration에는 initial delta가 유지된다.
                    delta = init_delta * (delta_decay ** (epoch + 1 - exploration))
                    print(delta)
                action = action + np.random.normal(0, delta, 84)
                
                action_1 = torch.from_numpy(action[:81]).view(3,3,3,3)
                action_2 = torch.from_numpy(action[81:])
                
                model_list[1].weight.data = action_1.clone().float().to(device)
                model_list[1].bias.data = action_2.clone().float().to(device)
                optimizer_model.zero_grad()
                
                next_state = model(current_state)
                output = classifier(next_state)
                loss = F.cross_entropy(output, ground_truth)
                loss.backward(retain_graph=True)
                optimizer_model.step()
                
                action_1 = model_list[1].weight.data.clone().view(-1)
                action_2 = model_list[1].bias.data.clone().view(-1)
                action = torch.cat((action_1, action_2))
                
                next_state = model(current_state)
                output = classifier(next_state)
                reward = - F.cross_entropy(output, ground_truth)
                
                if output.max(1, keepdim=True)[0] < threshold:
                    if state_idx + 1 == budget:
                        terminal = np.zeros((1))
                    else:
                        terminal = np.ones((1))
                else:
                    terminal = np.zeros((1))
                agent_memory.add(current_state, action, next_state, reward, terminal, state_idx)
                
                x[i] = next_state.detach().clone()
            
            output = classifier(x)
            done = output.max(1, keepdim=True)[0]
            
            if classifier_memory.n_entries < warmup:
                pass
            else:
                batch_x, batch_y = classifier_memory.sample(batch_size)
            
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.long().to(device)
                
                optimizer_classifier.zero_grad()
                result = classifier(batch_x)
                loss = criterion(result, batch_y)
                loss.backward()
                optimizer_classifier.step()
                
            if agent_memory.n_entries < warmup:
                pass
            else:
                ## ---- get sample ---- ##
                batch, index = agent_memory.sample(batch_size)
            
                batch = np.array(batch).transpose()
            
                current_state = np.vstack(batch[0])
                action = np.vstack(batch[1])
                next_state = np.vstack(batch[2])
                reward = np.vstack(batch[3])
                terminal = np.vstack(batch[4])
                index = np.array(index)
                
                current_state = torch.from_numpy(current_state).float().to(device)
                action = torch.from_numpy(action).float().to(device)
                next_state = torch.from_numpy(next_state).float().to(device)
                reward = torch.from_numpy(reward).float().to(device)
                terminal = torch.from_numpy(terminal).float().to(device)            
                index = torch.from_numpy(index).float().to(device).view(batch_size,1)
                
                ## ---- moving average ---- ##
                batch_mean_reward = reward.mean().item()
                if moving_average is None:
                    moving_average = batch_mean_reward
                else:
                    moving_average += moving_alpha * (batch_mean_reward - moving_average)
                reward = reward - moving_average
                
                ######## -------- Critic -------- ########            
                optimizer_critic.zero_grad()
                
                ## ---- get priority: td error ---- ##
                with torch.no_grad():
                    critic_action_next = actor_target([(index+1)/budget, next_state])
                    critic_Q_target_next = critic_target([(index+1)/budget, next_state, critic_action_next])
                
                critic_Q_target = reward + discount * critic_Q_target_next * terminal
                critic_Q_expected = critic([index/budget, current_state, action])
                
                ## ---- update network ---- ##
                critic_loss = F.mse_loss(critic_Q_expected, critic_Q_target).mean()
                critic_loss.backward()
                optimizer_critic.step()
            
                ######## -------- Actor  -------- ########
                optimizer_actor.zero_grad()
                
                ## ---- get Q-value ---- ##
                actor_action_expected = actor([index/budget, current_state])
                Q_value = critic([index/budget, current_state, actor_action_expected])
            
                ## ---- update network ---- ##
                actor_loss = -Q_value.mean()
                actor_loss.backward()
                optimizer_actor.step()    
            
                ################ ---------------- update target network ---------------- ################
                soft_update(actor_target, actor, tau)
                soft_update(critic_target, critic, tau)
            
            state_idx += 1
            if state_idx == budget:
                break

    accuracy = 0
    with torch.no_grad():
        correct = 0
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            state_idx = 0
        
            output = classifier(x)
            done = output.max(1, keepdim=True)[0]
            
            pred = output[(done >= threshold).reshape(-1)].argmax(1, keepdim=True)
            correct += pred.eq(y[(done >= threshold).reshape(-1)].view_as(pred)).sum().item()
            
            while (done < threshold).sum().item():
                x = x[(done < threshold).reshape(-1)]
                y = y[(done < threshold).reshape(-1)]
                state_idx_tensor = torch.Tensor([state_idx/budget]).to(device)
                
                for i in range(x.shape[0]):
                    current_state = x[i].view(1,3,32,32)
                    ground_truth = y[i].view(1)
                
                    action = actor([state_idx_tensor, current_state])
                    action_1 = action[:,:81].view(3,3,3,3)
                    action_2 = action[:,81:].view(3)
                
                    model_list[1].weight.data = action_1.clone().to(device)
                    model_list[1].bias.data = action_2.clone().to(device)
                
                    next_state = model(current_state)
                
                    x[i] = next_state.detach().clone()
            
                output = classifier(x)
                done = output.max(1, keepdim=True)[0]
                
                pred = output[(done >= threshold).reshape(-1)].argmax(1, keepdim=True)
                correct += pred.eq(y[(done >= threshold).reshape(-1)].view_as(pred)).sum().item()
                
                state_idx += 1
                
                if state_idx == budget:
                    break
                
        accuracy = correct / len(test_loader.dataset)
        print("[Accuracy: %f]" % accuracy)


'''

















