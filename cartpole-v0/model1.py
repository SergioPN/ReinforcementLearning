import tensorflow as tf
from tensorflow import keras
from keras.callbacks import TensorBoard
from Generacion_Partidas import PartidasRandom
from keras.utils import to_categorical
import numpy as np
from time import time



def BuildModel(input_size, outputs = 2):
    model = keras.Sequential([
        keras.layers.Dense(128, activation=tf.nn.relu, input_shape = (input_size, )),
        keras.layers.Dropout(0.8),
        keras.layers.Dense(256, activation=tf.nn.relu),
        keras.layers.Dropout(0.8),
        keras.layers.Dense(512, activation=tf.nn.relu),
        keras.layers.Dropout(0.8),
        keras.layers.Dense(256, activation=tf.nn.relu),
        keras.layers.Dropout(0.8),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dropout(0.8),
        keras.layers.Dense(outputs, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='categorical_crossentropy',
                  metrics = ["accuracy"])

    return model

def TrainModel(training_data, model=False):

    obs = np.array([obs for data in partidas_train for obs in data["observations"]])
    mov = np.array([obs for data in partidas_train for obs in data["movimientos"]])
    mov = to_categorical(mov)

    if not model:
        model = BuildModel(input_size = 4)

    tensorboard = TensorBoard(log_dir="logs/{}".format(time())) # tensorboard --logdir=logs


    model.fit(obs, mov,
              epochs = 50,
              callbacks=[tensorboard])

    return model


partidas_train = PartidasRandom(num_partidas = 5000)

model = TrainModel(partidas_train)

pwd
