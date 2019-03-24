"""
Este modelo entrena una red de capas densas conforme juega
"""

# %%
import tensorflow as tf
import gym
from tensorflow import keras
from keras.callbacks import TensorBoard
from Generacion_Partidas import PartidasRandom
from keras.utils import to_categorical
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from time import time
%matplotlib inline


def BuildModel(input_size, outputs = 2):
    model = keras.Sequential([
        keras.layers.Dense(32, activation=tf.nn.relu, input_shape = (input_size, )),
        keras.layers.Dropout(0.2),
        # keras.layers.Dense(64, activation=tf.nn.relu),
        # keras.layers.Dropout(0.2),
        # keras.layers.Dense(128, activation=tf.nn.relu),
        # keras.layers.Dropout(0.2),
        # keras.layers.Dense(64, activation=tf.nn.relu),
        # keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(outputs, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='categorical_crossentropy',
                  metrics = ["accuracy"])

    return model

# %%

model = BuildModel(4)

env = gym.make('CartPole-v0')
observation = env.reset()
env.render()
score = 0
record = 0
games = 0
while games < 1000:
    predict = model.predict(observation.reshape(1,4))
    action = np.argmax(predict)
    new_observation, reward, done, info = env.step(action)
    score += reward
    new_observation = new_observation.reshape(1,4)
    reward = to_categorical(reward, num_classes = 2).reshape(1,2).astype(int)
    history = model.fit(new_observation,
                        reward,
                        verbose = 0,
                        shuffle = False)
    observation = new_observation
    env.render()
    # print(history.history['loss'], end = "")
    if done:
        if score > record:
            print("Record: ", score)
            record = score
        observation = env.reset()
        score = 0
        games += 1

# env.close()
