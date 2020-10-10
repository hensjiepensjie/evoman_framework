#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 10:42:07 2020

@author: devinmulder
"""
 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

result_best_method11 = pd.DataFrame()
result_mean_method11 = pd.DataFrame()
result_std_method11 = pd.DataFrame()

result_best_method12 = pd.DataFrame()
result_mean_method12 = pd.DataFrame()
result_std_method12 = pd.DataFrame()

result_best_method21 = pd.DataFrame()
result_mean_method21 = pd.DataFrame()
result_std_method21 = pd.DataFrame()

result_best_method22 = pd.DataFrame()
result_mean_method22 = pd.DataFrame()
result_std_method22 = pd.DataFrame()

runs = 10
tekening = 1

if tekening == 1:
    for i in range(1,runs+1):
        file = '1467_multi_{}'.format(i)
        file_location = '1467_multi'
        
        data = pd.read_table(file_location + '/' + file + '/results.txt', delim_whitespace=True)
        result_best_method11[file] = data["best"]
        result_mean_method11[file] = data["mean"]
        result_std_method11[file] = data["std"] 

    for i in range(1,runs+1):
        file = '2358_multi_{}'.format(i)
        file_location = '2358_multi'
        
        data = pd.read_table(file_location + '/' + file + '/results.txt', delim_whitespace=True)
        result_best_method12[file] = data["best"]
        result_mean_method12[file] = data["mean"]
        result_std_method12[file] = data["std"]
    
    for i in range(1,runs+1):
        file = '1467_multi_{}'.format(i)
        file_location = '1467_multi_tournament'
        
        data = pd.read_table(file_location + '/' + file + '/results.txt', delim_whitespace=True)
        result_best_method21[file] = data["best"]
        result_mean_method21[file] = data["mean"]
        result_std_method21[file] = data["std"] 

    for i in range(1,runs+1):
        file = '2358_multi_{}'.format(i)
        file_location = '2358_multi_tournament'
        
        data = pd.read_table(file_location + '/' + file + '/results.txt', delim_whitespace=True)
        result_best_method22[file] = data["best"]
        result_mean_method22[file] = data["mean"]
        result_std_method22[file] = data["std"]
    
    
    result_best_method12['average'] = result_best_method12.mean(numeric_only=True, axis=1)   
    result_best_method12['std'] = result_best_method12.std(numeric_only=True, axis=1) 
    result_mean_method12['average'] = result_mean_method12.mean(numeric_only=True, axis=1) 
    result_std_method12['average'] = result_std_method12.mean(numeric_only=True, axis=1) 
    
    result_best_method11['average'] = result_best_method11.mean(numeric_only=True, axis=1) 
    result_best_method11['std'] = result_best_method11.std(numeric_only=True, axis=1)   
    result_mean_method11['average'] = result_mean_method11.mean(numeric_only=True, axis=1) 
    result_std_method11['average'] = result_std_method11.mean(numeric_only=True, axis=1) 
    
    result_best_method22['average'] = result_best_method22.mean(numeric_only=True, axis=1)   
    result_best_method22['std'] = result_best_method22.std(numeric_only=True, axis=1) 
    result_mean_method22['average'] = result_mean_method22.mean(numeric_only=True, axis=1) 
    result_std_method22['average'] = result_std_method22.mean(numeric_only=True, axis=1) 
    
    result_best_method21['average'] = result_best_method21.mean(numeric_only=True, axis=1) 
    result_best_method21['std'] = result_best_method21.std(numeric_only=True, axis=1)   
    result_mean_method21['average'] = result_mean_method21.mean(numeric_only=True, axis=1) 
    result_std_method21['average'] = result_std_method21.mean(numeric_only=True, axis=1) 

    f = plt.figure(figsize=(10,4))
    ax1 = f.add_subplot(121)
    ax2 = f.add_subplot(122)


    f.suptitle('Two Algorithms, Enemy 1467 VS Enemy 2358')
    ax1.plot(result_best_method11["average"],color='blue')
    ax1.plot(result_mean_method11["average"],color='green')
    ax1.fill_between(range(len(result_mean_method11["average"])), result_mean_method11["average"]-result_std_method11["average"], result_mean_method11["average"]+result_std_method11["average"], color='gray', alpha=0.4)
    ax1.fill_between(range(len(result_best_method11["average"])), result_best_method11["average"]-result_best_method11["std"], result_best_method11["average"]+result_best_method11["std"], color='gray', alpha=0.4)
    ax1.plot(result_best_method21["average"], color='red')
    ax1.plot(result_mean_method21["average"], color='orange')
    ax1.fill_between(range(len(result_mean_method21["average"])), result_mean_method21["average"]-result_std_method21["average"], result_mean_method21["average"]+result_std_method21["average"], color='gray', alpha=0.4)
    ax1.fill_between(range(len(result_best_method21["average"])), result_best_method21["average"]-result_best_method21["std"], result_best_method21["average"]+result_best_method21["std"], color='gray', alpha=0.4)
    
    ax2.plot(result_best_method12["average"],color='blue')
    ax2.plot(result_mean_method12["average"],color='green')
    ax2.fill_between(range(len(result_mean_method12["average"])), result_mean_method12["average"]-result_std_method12["average"], result_mean_method12["average"]+result_std_method12["average"], color='gray', alpha=0.4)
    ax2.fill_between(range(len(result_best_method12["average"])), result_best_method12["average"]-result_best_method12["std"], result_best_method12["average"]+result_best_method12["std"], color='gray', alpha=0.4)
    ax2.plot(result_best_method22["average"], color='red')
    ax2.plot(result_mean_method22["average"], color='orange')
    ax2.fill_between(range(len(result_mean_method22["average"])), result_mean_method22["average"]-result_std_method22["average"], result_mean_method22["average"]+result_std_method22["average"], color='gray', alpha=0.4)
    ax2.fill_between(range(len(result_best_method22["average"])), result_best_method22["average"]-result_best_method22["std"], result_best_method22["average"]+result_best_method22["std"], color='gray', alpha=0.4)



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
   
n_hidden_neurons= 10
    
gainscores1 = []
#get the new alogirhm
for i in range(1,runs+1):
    file = '1467_multi_{}'.format(i)
    file_location = '1467_multi_tournament'
    experiment_name = (file_location + '/' + file)
    gainTotal = 0
        
    env = sim_environment(experiment_name, [1,2,3,4,5,6,7,8], n_hidden_neurons)
    bsol = np.loadtxt(file_location + '/' + file + '/best.txt')
    f,p,e,t = env.play(pcont=bsol)
    gain1 = p - e
    gainscores1.append(gain1)
        
        
        
        
        
        
        
        
        
#boxplots

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

A = [[-10.68,-9.59,-24.62,-16.65,-36.03,-26.39,-35.02,-11.42,2.38,-4.87],  [-37.12,-26.49,-24.8,-37.29,-22.39,-40.66,-17.51,-17.26,-14.73,-11.04]]
B = [[-9.98,-14.05,2.04,3.33,-6.51,-9.53,-9.20,-8.37,-8.65,-9.43],  [-12.35,-8.62,5.65,-3.51,-5.35,-4.28,-7.30,-12.3,-4.93,-9.80]]

df = pd.DataFrame({'Enemy':['2','2','2','2','2','2','2','2','2','2', '3','3','3','3','3','3','3','3','3','3'],\
                   'alg1':[-10.68,-9.59,-24.62,-16.65,-36.03,-26.39,-35.02,-11.42,2.38,-4.87,-9.98,-14.05,2.04,3.33,-6.51,-9.53,-9.20,-8.37,-8.65,-9.43],\
                   'alg2':[-37.12,-26.49,-24.8,-37.29,-22.39,-40.66,-17.51,-17.26,-14.73,-11.04,-12.35,-8.62,5.65,-3.51,-5.35,-4.28,-7.30,-12.3,-4.93,-9.80]})

df = df[['Enemy','alg1','alg2']]

dd=pd.melt(df,id_vars=['Enemy'],value_vars=['alg1','alg2'],var_name='alg')
sns.boxplot(x='Enemy',y='value',data=dd,hue='alg')


from scipy.stats import ttest_ind, ttest_ind_from_stats

A1 = [-10.68,-9.59,-24.62,-16.65,-36.03,-26.39,-35.02,-11.42,2.38,-4.87]
A2 = [-37.12,-26.49,-24.8,-37.29,-22.39,-40.66,-17.51,-17.26,-14.73,-11.04]
B1 = [-9.98,-14.05,2.04,3.33,-6.51,-9.53,-9.20,-8.37,-8.65,-9.43]
B2 = [-12.35,-8.62,5.65,-3.51,-5.35,-4.28,-7.30,-12.3,-4.93,-9.80]

t, p = ttest_ind(B1, B2, equal_var=False)
print(t)
print(p)



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
   
n_hidden_neurons= 10
    
gainscores1 = []
file = '2358_multi_10'
file_location = '2358_multi_tournament'
#get the new alogirhm
for i in range(1,6):
    
    experiment_name = (file_location + '/' + file)
    gainTotal = 0
        
    env = sim_environment(experiment_name, [1,2,3,4,5,6,7,8], n_hidden_neurons)
    bsol = np.loadtxt(file_location + '/' + file + '/final_sol.txt')
    f,p,e,t = env.play(pcont=bsol)
    gain1 = p - e
    gainscores1.append(gain1)

