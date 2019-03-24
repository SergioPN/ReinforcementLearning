"""
Este modelo entrena una red de capas densas
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



def BuildModel(input_size, outputs = 2):
    model = keras.Sequential([
        keras.layers.Dense(32, activation=tf.nn.relu, input_shape = (input_size, )),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(outputs, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='categorical_crossentropy',
                  metrics = ["accuracy"])

    return model

def TrainModel(training_data):

    obs = np.array([obs for data in partidas_train for obs in data["observations"]])
    mov = np.array([obs for data in partidas_train for obs in data["movimientos"]])
    mov = to_categorical(mov)

    # if not model:
    #     print("Starting a new model")
    #     model = BuildModel(input_size = 4)

    # tensorboard = TensorBoard(log_dir="logs/{}".format(time())) # tensorboard --logdir=logs

    model.fit(obs, mov,
              epochs = 50,
              batch_size = 128,
              callbacks=[])

    return model


partidas_train = PartidasRandom(num_partidas = 5000000, score_minimo = 180)


model = TrainModel(partidas_train)


model.save("./models/test1")



def PlayModel(model):
    env = gym.make('CartPole-v0')
    record = 0
    for _ in range(100):
        done = False
        observation = env.reset()
        env.render()
        current_points = 0
        for i in range(500):
            action = np.argmax(model.predict(observation.reshape((1,4))))
            # if action==1:
            #     action = 0
            # else:
            #     action = 1
            observation, reward, done, info = env.step(action)
            current_points += 1
            env.render()

            if done:
                if current_points > record:
                    print(f"Record with {current_points}")
                    record = current_points
                break
