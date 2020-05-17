import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import gym
from collections import deque
import numpy as np
from dataclasses import dataclass
import random
import matplotlib.pyplot as plt
from typing import Any

@dataclass
class Sarsd:
    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool

class ReplayBuffer:
    def __init__(self, buffer_size=1000):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=self.buffer_size)
    def insert(self, sarsd):
        self.buffer.append(sarsd)
    def sample(self, num_samples):
        return random.sample(self.buffer, num_samples)

class Model(nn.Module):
    def __init__(self, state_size, action_size):
        super(Model, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.net = nn.Sequential(
                                 nn.Linear(in_features = state_size, out_features=24),
                                 nn.ReLU(),
                                 nn.Linear(in_features = 24, out_features = 24),
                                 nn.ReLU(),
                                 nn.Linear(in_features = 24, out_features = action_size)
        )

        self.opt = optim.Adam(self.net.parameters(), lr=1e-4)

    def forward(self, x):
        return self.net(x)

    def train_step(self, memory, num_actions):
        states = torch.stack(torch.Tensor([s.state for s in memory]))
        actions = torch.stack(torch.Tensor([s.action for s in memory]))
        rewards = torch.stack(torch.Tensor([s.reward for s in memory]))
        next_states = torch.stack(torch.Tensor([s.next_state for s in memory]))
        donetes = torch.stack(torch.Tensor([0 if s.done else 1 for s in memory]))

        with torch.no_grad():
            q_vals_next = self.net.eval(next_states)

        self.net.optimizer.zero_grad()
        q_vals = self.net(states)
        one_hote_actions = F.one_hot(torch.LongTensor(actions, num_actions))

        loss = (rewards + donetes * q_vals - torch.sum(q_vals * one_hote_actions, -1)).mean()




class DQNAgent():
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95

        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 1e-3

        self.model = self._build_model()

    def _build_model(self):
        return Model(self.state_size, self.action_size)

    def actua(self, state):
        if np.random.random() < self.epsilon:
            action = random.randrange(self.action_size)
        else:
            q_vals = self.model(state)
            action = q_vals.max(-1)
        return action

    def replay(self, batch_size):

        minibatch = random.sample(self.memory, batch_size) # Porque no toda la memoria ?

        for state, action, reward, new_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(new_state)[0])) # reward + discout * futuro max expected value

            target_f = self.model.predict(state) #expected value que predice el modelo ahora
            target_f[0][action] = target #Para este estado, la accion tomada me ha dado el reward
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, path):
        self.model.load_weights(path)
    def save(self, path):
        self.model.save_weights(path)

#%%
try:
    env.close()
except:
    pass
env = gym.make('CartPole-v0')

agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
buffer = ReplayBuffer(buffer_size = 1000)
