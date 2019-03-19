import gym
import random
import numpy as np
from tqdm import tqdm
env = gym.make('CartPole-v0')
env.reset()


def PartidasRandom(num_partidas = 2000, score_minimo = 50):
    partidas_train = []
    for _ in range(num_partidas):
        done = False
        observation = env.reset()
        movimientos_partida = []
        observations_partida = []
        score_partida = 0
        while not done: #Jugando la partida
            observations_partida.append(observation)
            action = random.randint(0,1)
            observation, reward, done, info = env.step(action)
            movimientos_partida.append(action)
            score_partida += reward


        if score_partida >= score_minimo:
            partidas_train.append({"observations": observations_partida,
                                   "movimientos": movimientos_partida,
                                   "score": score_partida})
    return partidas_train


partidas_train = PartidasRandom(num_partidas = 50000)
