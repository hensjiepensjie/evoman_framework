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

A = [[6.000000000000604, -136.9999999999995, -129.59999999999948, -89.7999999999997, -79.19999999999939, -237.79999999999973, -142.9999999999995, -162.79999999999967, -105.59999999999921, -235.5999999999995],  [-227.39999999999966, -82.59999999999944, -132.3999999999994, -171.5999999999998, -134.39999999999966, -157.39999999999947, -82.59999999999971, -128.59999999999968, -157.59999999999968, 69.60000000000066]]
B = [[-123.79999999999967, -134.19999999999973, 28.600000000000982, -75.39999999999942, -229.39999999999972, -130.59999999999945, -208.39999999999952, -36.19999999999944, -312.0, -74.39999999999966],  [-196.59999999999948, -109.19999999999968, -130.19999999999948, -73.59999999999943, -54.79999999999939, -103.39999999999948, -76.99999999999945, -165.9999999999997, -113.59999999999943, -31.999999999999446]]

df = pd.DataFrame({'Enemy':['2','2','2','2','2','2','2','2','2','2', '3','3','3','3','3','3','3','3','3','3'],\
                   'alg1':[6.000000000000604, -136.9999999999995, -129.59999999999948, -89.7999999999997, -79.19999999999939, -237.79999999999973, -142.9999999999995, -162.79999999999967, -105.59999999999921, -235.5999999999995,-123.79999999999967, -134.19999999999973, 28.600000000000982, -75.39999999999942, -229.39999999999972, -130.59999999999945, -208.39999999999952, -36.19999999999944, -312.0, -74.39999999999966],\
                   'alg2':[-227.39999999999966, -82.59999999999944, -132.3999999999994, -171.5999999999998, -134.39999999999966, -157.39999999999947, -82.59999999999971, -128.59999999999968, -157.59999999999968, 69.60000000000066,-196.59999999999948, -109.19999999999968, -130.19999999999948, -73.59999999999943, -54.79999999999939, -103.39999999999948, -76.99999999999945, -165.9999999999997, -113.59999999999943, -31.999999999999446]})

df = df[['Enemy','alg1','alg2']]

dd=pd.melt(df,id_vars=['Enemy'],value_vars=['alg1','alg2'],var_name='alg')
sns.boxplot(x='Enemy',y='value',data=dd,hue='alg' )


from scipy.stats import ttest_ind, ttest_ind_from_stats

A1 = [6.000000000000604, -136.9999999999995, -129.59999999999948, -89.7999999999997, -79.19999999999939, -237.79999999999973, -142.9999999999995, -162.79999999999967, -105.59999999999921, -235.5999999999995]
A2 = [-227.39999999999966, -82.59999999999944, -132.3999999999994, -171.5999999999998, -134.39999999999966, -157.39999999999947, -82.59999999999971, -128.59999999999968, -157.59999999999968, 69.60000000000066]
B1 = [-123.79999999999967, -134.19999999999973, 28.600000000000982, -75.39999999999942, -229.39999999999972, -130.59999999999945, -208.39999999999952, -36.19999999999944, -312.0, -74.39999999999966]
B2 = [-196.59999999999948, -109.19999999999968, -130.19999999999948, -73.59999999999943, -54.79999999999939, -103.39999999999948, -76.99999999999945, -165.9999999999997, -113.59999999999943, -31.999999999999446]

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

