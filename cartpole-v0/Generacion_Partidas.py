import gym
import random
env = gym.make('CartPole-v0')
env.reset()


num_partidas = 100
score_minimo = 50
partidas = []


for _ in range(num_partidas):
    done = False
    cur_observation = env.reset()
    observations_partida = [cur_observation]
    movimientos_partida = []
    scores_partida = []
    while not done: #Jugando al partida
        action = random.randint(0,1)
        observation, reward, done, info = env.step(action)
        movimientos_partida.append(action)
        scores_partida.append(reward)
