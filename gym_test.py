import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
%matplotlib inline

#%%
env = gym.make('CartPole-v0')
env.reset()
#%%
dict = {}
dict["reward"] = []
dict["observation"] = []
for ii in range(1000):
    env.render()
    # env.step(env.action_space.sample()) # take a random action
    # action = env.action_space.sample()
    # action = 0
    observation, reward, done, info = env.step(action)
    if observation[0] < 0:
        action = 1
    else:
        action = 0
    dict["reward"].append(reward)
    dict["observation"].append(observation)



df = pd.DataFrame(np.vstack(dict["observation"]))
df[1].plot()

#%%
env.close()
