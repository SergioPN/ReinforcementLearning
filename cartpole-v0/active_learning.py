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


class Agent(input_size, outputs = 2):
    def __init__(self):
        pass

    def Policy(state):
        """
        Este es el que dado un estado devuelve una accion
        Policy(a|s)=A
        """

    def ValueFunction(state):
        """
        Este es el encargado de encontrar un policy que maximice el reward
        V(s) = E(R)
        """



# %%

model = BuildModel(4)

enviroment = gym.make('CartPole-v0')
state = enviroment.reset()
# enviroment.render()
score = 0
record = 0
games = 0
while games < 1000:
    predict = model.predict(state.reshape(1,4))
    action = np.argmax(predict)
    new_state, reward, done, info = enviroment.step(action)
    score += reward
    new_state = new_state.reshape(1,4)
    action = to_categorical(action, num_classes = 2).reshape(1,2).astype(int)
    history = model.fit(state,
                        np.round(predict),
                        verbose = 0,
                        shuffle = False)
    state = new_state
    # enviroment.render()
    # print(history.history['loss'], end = "")
    if done:
        if score > record:
            print("Record: ", score)
            record = score
        state = enviroment.reset()
        score = 0
        games += 1

# enviroment.close()
