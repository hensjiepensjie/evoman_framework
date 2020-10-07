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

from demo_controller import player_controller
import numpy as np
from environment import Environment
enemy = 2
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

result_best_new2 = []
result_mean_new2= []
result_std_new2= []
result_best_old2= []
result_mean_old2= []
result_std_old2= []
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
result_best_new['std'] = result_best_new.std(numeric_only=True, axis=1) 
result_mean_new['average'] = result_mean_new.mean(numeric_only=True, axis=1) 
result_std_new['average'] = result_std_new.mean(numeric_only=True, axis=1) 

result_best_old['average'] = result_best_old.mean(numeric_only=True, axis=1) 
result_best_old['std'] = result_best_old.std(numeric_only=True, axis=1)   
result_mean_old['average'] = result_mean_old.mean(numeric_only=True, axis=1) 
result_std_old['average'] = result_std_old.mean(numeric_only=True, axis=1) 

#enemy4
#result_best_new4 = result_best_new 
#result_mean_new4 = result_mean_new 
#result_std_new4 = result_std_new 
#result_best_old4 = result_best_old   
#result_mean_old4 = result_mean_old 
#result_std_old4 = result_std_old 

#enemy3
#result_best_new3 = result_best_new 
#result_mean_new3 = result_mean_new 
#result_std_new3 = result_std_new 
#result_best_old3 = result_best_old   
#result_mean_old3 = result_mean_old 
#result_std_old3 = result_std_old 

#enemy2
#result_best_new2 = result_best_new 
#result_mean_new2 = result_mean_new 
#result_std_new2 = result_std_new 
#result_best_old2 = result_best_old   
#result_mean_old2 = result_mean_old 
#result_std_old2 = result_std_old 


f = plt.figure(figsize=(10,3))
ax1 = f.add_subplot(131)
ax2 = f.add_subplot(132)
ax3 = f.add_subplot(133)


f.suptitle('Best and mean fitness for enemy 2, 3 and 4.')
ax1.plot(result_best_new2["average"])
ax1.plot(result_best_old2["average"])
ax1.plot(result_mean_new2["average"])
ax1.plot(result_mean_old2["average"])
ax1.fill_between(range(len(result_mean_new2["average"])), result_mean_new2["average"]-result_std_new2["average"]/5.5, result_mean_new2["average"]+result_std_new2["average"]/5.5, color='gray', alpha=0.4)
ax1.fill_between(range(len(result_mean_old2["average"])), result_mean_old2["average"]-result_std_old2["average"]/5.5, result_mean_old2["average"]+result_std_old2["average"]/5.5, color='gray', alpha=0.4)
ax1.fill_between(range(len(result_best_new2["average"])), result_best_new2["average"]-result_best_new2["std"]/5.5, result_best_new2["average"]+result_best_new2["std"]/5.5, color='gray', alpha=0.4)
ax1.fill_between(range(len(result_best_old2["average"])), result_best_old2["average"]-result_best_old2["std"]/5.5, result_best_old2["average"]+result_best_old2["std"]/5.5, color='gray', alpha=0.4)


ax2.plot(result_best_new3["average"])
ax2.plot(result_best_old3["average"])
ax2.plot(result_mean_new3["average"])
ax2.plot(result_mean_old3["average"])
ax2.fill_between(range(len(result_mean_new3["average"])), result_mean_new3["average"]-result_std_new3["average"]/5.5, result_mean_new3["average"]+result_std_new3["average"]/5.5, color='gray', alpha=0.4)
ax2.fill_between(range(len(result_mean_old3["average"])), result_mean_old3["average"]-result_std_old3["average"]/5.5, result_mean_old3["average"]+result_std_old3["average"]/5.5, color='gray', alpha=0.4)
ax2.fill_between(range(len(result_best_new3["average"])), result_best_new3["average"]-result_best_new3["std"]/5.5, result_best_new3["average"]+result_best_new3["std"]/5.5, color='gray', alpha=0.4)
ax2.fill_between(range(len(result_best_old3["average"])), result_best_old3["average"]-result_best_old3["std"]/5.5, result_best_old3["average"]+result_best_old3["std"]/5.5, color='gray', alpha=0.4)



ax3.plot(result_best_new4["average"])
ax3.plot(result_best_old4["average"])
ax3.plot(result_mean_new4["average"])
ax3.plot(result_mean_old4["average"])
ax3.fill_between(range(len(result_mean_new4["average"])), result_mean_new4["average"]-result_std_new4["average"]/5.5, result_mean_new4["average"]+result_std_new4["average"]/5.5, color='gray', alpha=0.4)
ax3.fill_between(range(len(result_mean_old4["average"])), result_mean_old4["average"]-result_std_old4["average"]/5.5, result_mean_old4["average"]+result_std_old4["average"]/5.5, color='gray', alpha=0.4)
ax3.fill_between(range(len(result_best_new4["average"])), result_best_new4["average"]-result_best_new4["std"]/5.5, result_best_new4["average"]+result_best_new4["std"]/5.5, color='gray', alpha=0.4)
ax3.fill_between(range(len(result_best_old4["average"])), result_best_old4["average"]-result_best_old4["std"]/5.5, result_best_old4["average"]+result_best_old4["std"]/5.5, color='gray', alpha=0.4)



#fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
#fig.suptitle('Horizontally stacked subplots')

#ax1.plot(result_best_new["average"])
#ax1.plot(result_best_old["average"])
#ax1.ylabel('best Fitness')
#ax1.xlabel('gens')
#plt.show()

#ax1.plot(result_mean_new["average"])
#ax1.plot(result_mean_old["average"])
#ax1.fill_between(range(len(result_mean_new["average"])), result_mean_new["average"]-result_std_new["average"], result_mean_new["average"]+result_std_new["average"], color='gray', alpha=0.4)
#ax1.fill_between(range(len(result_mean_old["average"])), result_mean_old["average"]-result_std_old["average"], result_mean_old["average"]+result_std_old["average"], color='gray', alpha=0.4)


#ax2.ylabel('mean Fitness')
#ax2.xlabel('gens')
#plt.show()

#plt.plot(result_std_new["average"])
#plt.plot(result_std_old["average"])
#ax3.ylabel('std Fitness')
#ax3.xlabel('gens')
#plt.show()

 
#plt.fill_between(range(len(result_best["average"])), result_best["average"]-result_std["average"], result_best["average"]+result_std["average"], color='gray', alpha=0.4)


A = [[42,35,27,32,29,29,46,78,82,78],  [84,82,82,86,86,76,82,82,84,84]]
B = [[34,32,48,30,38,12,34,14,26,82],  [-20,40,22,32,12,32,2,20,42,48]]

plt.boxplot(A)


df = pd.DataFrame({'Enemy':['2','2','2','2','2','2','2','2','2','2', '3','3','3','3','3','3','3','3','3','3','4','4','4','4','4','4','4','4','4','4'],\
                   'alg2':[88,76,88,78,78,78,90,78,82,78,34,32,48,30,38,12,34,14,26,82,62,-40,-30,47,40,39,58,12,67,-10],\
                   'alg1':[84,82,82,86,86,76,82,82,84,84,-20,40,22,32,12,32,2,20,42,48,-40,39,38,-30,9,9,-40,6,59,-16]})

df = df[['Enemy','alg2','alg1']]

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
dd=pd.melt(df,id_vars=['Enemy'],value_vars=['alg2','alg1'],var_name='algorithm')
sns.boxplot(x='Enemy',data=dd,hue='algorithm')


from scipy.stats import ttest_ind, ttest_ind_from_stats

A1 = [88,76,88,78,78,78,90,78,82,78]
A2 = [84,82,82,86,86,76,82,82,84,84]
B1 = [34,32,48,30,38,12,34,14,26,82] 
B2 = [-20,40,22,32,12,32,2,20,42,48]
C1 = [62,-40,-30,47,40,39,58,12,67,-10]
C2 = [-40,39,38,-30,9,9,-40,6,59,-16]

t, p = ttest_ind(C1, C2, equal_var=False)
print(t)
print(p)





