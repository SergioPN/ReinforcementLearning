import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
import gym
from collections import deque
import numpy as np
import random

# CartPole-v0 is considered "solved" when the agent obtains an average reward of at least 195.0 over 100 consecutive episodes.


class DQNAgent():
    def __init__(self, state_size, action_size):
        self.memory = deque(maxlen=2000)
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95

        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 1e-3

        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim = self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), loss='mse')
        return model

    def actua(self, state):
        if np.random.random() < self.epsilon:
            action = random.randrange(self.action_size)
        else:
            action = self.model.predict(state)[0].argmax()
        return action

    def remember(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

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

done = False
n_episodes = 2000
batch_size = 32
import time
for e in range(n_episodes):
    state = env.reset().reshape(1, -1)

    for step in range(300):
        # env.render()
        # time.sleep(0.1)
        action = agent.actua(state)
        new_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        new_state = new_state.reshape(1, -1)


        agent.remember(state, action, reward, new_state, done)

        state = new_state

        if done:
            print(f"{e}/{n_episodes}, score: {step}, e:{agent.epsilon}")
            break

    if len(agent.memory) > batch_size:
        agent.replay(batch_size)
