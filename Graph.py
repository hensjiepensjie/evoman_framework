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
result_best_new = pd.DataFrame()
result_mean_new = pd.DataFrame()
result_std_new = pd.DataFrame()
result_best_old = pd.DataFrame()
result_mean_old = pd.DataFrame()
result_std_old = pd.DataFrame()
gainscores_new = []
gainscores_old = []

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

#get the new alogirhm
#for i in range(1,runs+1):
#    file = 'test{}.{}'.format(enemy, i)
#    file_location = 'enemy_{}_new30'.format(enemy)
#    experiment_name = (file_location + '/' + file)
#    gainTotal = 0
    
    
    #for j in range(1,6):
#    env = sim_environment(experiment_name, enemy, n_hidden_neurons)
#    bsol = np.loadtxt(file_location + '/' + file + '/best.txt')
#    f,p,e,t = env.play(pcont=bsol)
#    gain1 = p - e
    #gainTotal = gainTotal + gain1
    #print(gainTotal)
#    gainscores_new.append(gain1)
    #gainscores_new.append(gainTotal/5)

#get the old algorithm
#for i in range(1,runs+1):
#    file = 'test{}.{}'.format(enemy, i)
#    file_location = 'enemy_{}_old30'.format(enemy)
#    experiment_name = (file_location + '/' + file)
#    gainTotal = 0
    
    
    #for j in range(1,6):
#    env = sim_environment(experiment_name, enemy, n_hidden_neurons)
#    bsol = np.loadtxt(file_location + '/' + file + '/best.txt')
#    f,p,e,t = env.play(pcont=bsol)
#    gain1 = p - e
    #gainTotal = gainTotal + gain1
    #print(gainTotal)
#    gainscores_old.append(gain1)
    #gainscores_new.append(gainTotal/5)

#plt.boxplot(gainscores_new)
#plt.boxplot(gainscores_old)
#sys.exit(0)

for i in range(1,runs+1):
    file = 'test{}.{}'.format(enemy, i)
    file_location = 'enemy_{}_new30'.format(enemy)

    data = pd.read_table(file_location + '/' + file + '/results.txt', delim_whitespace=True)
    result_best_new[file] = data["best"]
    result_mean_new[file] = data["mean"]
    result_std_new[file] = data["std"] 

for i in range(1,runs+1):
    file = 'test{}.{}'.format(enemy, i)
    file_location = 'enemy_{}_old30'.format(enemy)

    data = pd.read_table(file_location + '/' + file + '/results.txt', delim_whitespace=True)
    result_best_old[file] = data["best"]
    result_mean_old[file] = data["mean"]
    result_std_old[file] = data["std"] 




result_best_new['average'] = result_best_new.mean(numeric_only=True, axis=1)   
result_mean_new['average'] = result_mean_new.mean(numeric_only=True, axis=1) 
result_std_new['average'] = result_std_new.mean(numeric_only=True, axis=1) 

result_best_old['average'] = result_best_old.mean(numeric_only=True, axis=1)   
result_mean_old['average'] = result_mean_old.mean(numeric_only=True, axis=1) 
result_std_old['average'] = result_std_old.mean(numeric_only=True, axis=1) 


plt.plot(result_best_new["average"])
plt.plot(result_best_old["average"])
plt.fill_between(range(len(result_best_new["average"])), result_best_new["average"]-result_std_new["average"], result_best_new["average"]+result_std_new["average"], color='gray', alpha=0.4)
plt.fill_between(range(len(result_best_old["average"])), result_best_old["average"]-result_std_old["average"], result_best_old["average"]+result_std_old["average"], color='gray', alpha=0.4)
plt.ylabel('best Fitness')
plt.xlabel('gens')
plt.show()

plt.plot(result_mean_new["average"])
plt.plot(result_mean_old["average"])
plt.ylabel('mean Fitness')
plt.xlabel('gens')
plt.show()

plt.plot(result_std_new["average"])
plt.plot(result_std_old["average"])
plt.ylabel('std Fitness')
plt.xlabel('gens')
plt.show()

 
#plt.fill_between(range(len(result_best["average"])), result_best["average"]-result_std["average"], result_best["average"]+result_std["average"], color='gray', alpha=0.4)


#A = [[88,76,88,78,78,78,90,78,82,78],  [84,82,82,86,86,76,82,82,84,84]]
#B = [[34,32,48,30,38,12,34,14,26,82],  [-20,40,22,32,12,32,2,20,42,48]]
#C = [[62,-40,-30,47,40,39,58,12,67,-10], [-40,39,38,-30,9,9,-40,6,59,-16]]

#plt.boxplot(A)


#df = pd.DataFrame({'Enemy':['2','2','2','2','2','2','2','2','2','2', '3','3','3','3','3','3','3','3','3','3','4','4','4','4','4','4','4','4','4','4'],\
 #                 'alg1':[88,76,88,78,78,78,90,78,82,78,34,32,48,30,38,12,34,14,26,82,62,-40,-30,47,40,39,58,12,67,-10],\
#                  'alg2':[84,82,82,86,86,76,82,82,84,84,-20,40,22,32,12,32,2,20,42,48,-40,39,38,-30,9,9,-40,6,59,-16]})

#df = df[['Enemy','alg1','alg2']]

#import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
#dd=pd.melt(df,id_vars=['Enemy'],value_vars=['alg1','alg2'],var_name='algorithm')
#sns.boxplot(x='Enemy',y='Gain',data=dd,hue='Algorithm')













