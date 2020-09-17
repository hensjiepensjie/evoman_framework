#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 12:13:52 2020

@author: devinmulder
"""
import sys, os
sys.path.insert(0, 'evoman') 
import pandas as pd
import matplotlib.pyplot as plt
from environment import Environment
from demo_controller import player_controller
import numpy as np

enemy = 4
runs = 10
n_hidden_neurons = 10
result_best = pd.DataFrame()
result_mean = pd.DataFrame()
result_std = pd.DataFrame()
gainscores = []


def sim_environment(experiment_name, enemy, n_hidden_neurons):
    # initializes environment with ai player using random controller, playing against static enemy
    env = Environment(experiment_name=experiment_name,
                      enemies=[enemy],  # Change enemies in top of file
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest")

    return env


def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

#for i in range(1,runs+1):
#    file = 'test{}.{}'.format(enemy, i)
#    file_location = 'enemy_{}'.format(enemy)
#    experiment_name = (file_location + '/' + file)
#    gainTotal = 0
    
    
#    for j in range(1,6):
#        env = sim_environment(experiment_name, enemy, n_hidden_neurons)
#        bsol = np.loadtxt(file_location + '/' + file + '/best.txt')
#        f,p,e,t = env.play(pcont=bsol)
#        gain1 = p - e
#        gainTotal = gainTotal + gain1
#        print(gainTotal)
    
#    gainscores.append(gainTotal/5)

#plt.boxplot(gainscores)

#sys.exit(0)

for i in range(1,runs+1):
    file = 'test{}.{}'.format(enemy, i)
    file_location = 'enemy_{}'.format(enemy)

    data = pd.read_table(file_location + '/' + file + '/results.txt', delim_whitespace=True)
    result_best[file] = data["best"]
    result_mean[file] = data["mean"]
    result_std[file] = data["std"] 

result_best['average'] = result_best.mean(numeric_only=True, axis=1)   
result_mean['average'] = result_mean.mean(numeric_only=True, axis=1) 
result_std['average'] = result_std.mean(numeric_only=True, axis=1) 

plt.plot(result_best["average"])
plt.fill_between(range(len(result_best["average"])), result_best["average"]-result_std["average"], result_best["average"]+result_std["average"], color='gray', alpha=0.4)
plt.ylabel('best Fitness')
plt.xlabel('')
plt.show()

plt.plot(result_mean["average"])
plt.ylabel('mean Fitness')
plt.xlabel('')
plt.show()

plt.plot(result_std["average"])
plt.ylabel('std Fitness')
plt.xlabel('')
plt.show()

 



