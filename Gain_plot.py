import sys, os
sys.path.insert(0, 'evoman')
from demo_controller import player_controller
import numpy as np
from environment import Environment


def sim_environment(experiment_name, enemies, n_hidden_neurons):
    # initializes simulation in multi evolution mode, for multiple static enemies.
    env = Environment(experiment_name=experiment_name,
                      enemies=enemies,
                      multiplemode="yes",
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest")

    return env


n_hidden_neurons = 10

gainscores1 = []
file = '2358_multi_10'
file_location = '2358_multi_tournament'
# get the new algorithm
for i in range(1, 6):
    experiment_name = (file_location + '/' + file)
    gainTotal = 0

    env = sim_environment(experiment_name, [1, 2, 3, 4, 5, 6, 7, 8], n_hidden_neurons)
    bsol = np.loadtxt(file_location + '/' + file + '/final_sol.txt')
    f, p, e, t = env.play(pcont=bsol)

    file_aux = open(file_location + '/' + file + "/evoman_logs.txt", "r")

    for line in file_aux:
        if line.startswith('RUN:'):
            line = line.rsplit('; ')
            p_life = float(line[2].rsplit(': ')[1])
            e_life = float(line[3].rsplit(': ')[1])
            gain = p_life - e_life
            gainTotal += gain

    file_aux.close()

    gainscores1.append(gainTotal)

print(gainscores1)