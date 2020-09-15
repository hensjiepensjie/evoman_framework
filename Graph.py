#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 12:13:52 2020

@author: devinmulder
"""

import pandas as pd
import matplotlib.pyplot as plt

enemy = 4
runs = 10
result_best = pd.DataFrame()
result_mean = pd.DataFrame()
result_std = pd.DataFrame()


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





