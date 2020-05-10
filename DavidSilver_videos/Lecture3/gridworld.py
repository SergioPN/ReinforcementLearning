import matplotlib.pyplot as plt
import numpy as np
from numba import njit


"""
Ejemplo del video https://www.youtube.com/watch?v=Nd1-UUMVfz4&t=1476s
Se trata de la actualización de V para la policy random.

Policy random es un movimiento aleatorio a una de las cuatro direcciones con
reward -1.

Al algoritmo actualiza el state value hasta cumplir cierto threshold
"""

#%%
@njit
def update_state(current_state, vk):
    reward_policy = -1
    i, j = current_state
    contador = 0

    if (i == j == 0) or (i == j == (N-1)):
        return 0

    for ii, jj in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
        if (ii == N) or (ii == -1) or (jj == N) or (jj == -1):
            contador += reward_policy+vk[i,j]
        else:
            contador += vk[ii, jj] + reward_policy

    return contador/4

N = 16
vk = np.zeros((N, N), dtype=np.float32)
updates = 0
#%%
while True:
    new_vk = np.zeros((N, N), dtype=np.float32)
    for i, row in enumerate(vk):
        for j, col in enumerate(row):
            new_vk[i, j] = update_state((i,j), vk)
    if np.sum(vk - new_vk)/N < 0.1:
        break
    vk = new_vk
    updates += 1

plt.imshow(vk, cmap="Spectral")
plt.colorbar()
plt.title(str(updates))
#%%
