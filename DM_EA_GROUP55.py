################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os
sys.path.insert(0, 'evoman') 
from environment import Environment
from demo_controller import player_controller
import matplotlib.pyplot as plt

# imports other libs
import time
import numpy as np
import random
from math import fabs,sqrt
import glob, os

experiment_name = 'test3.2'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

#number of....
n_hidden_neurons = 10

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name,
                  enemies=[4],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")


# default environment fitness is assumed for experiment
env.state_to_log() # checks environment state

ini = time.time()  # sets time marker

#run type train or test 
run_mode = 'train' # train or test

# number of weights for multilayer with 10 hidden neurons
number_of_weights = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

#Var pop and NN
dom_u = 1 #upperbound NN value
dom_l = -1 #lowerbound NN value
npop = 20 #population size       #if changed check parent selection
gens = 10 #number of generations
mutation_prob = 0.25
######################function definitions########################3

#simulate with a specific NN
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))

def parent_selection_roulette(fit_pop):
    

    max = sum(fit_pop.values())
    pick = 0
    pick = np.uniform(0, max)
    current = 0
    for key, value in fit_pop.items():
        current += value
        if current > pick:
            return key

def evolution(pop, fit_pop):
    
    parent1 = -1
    parent2 = -2
    
    partkilled = int(npop/2)  # a quarter of the population
    order = np.argsort(fit_pop)
    partchanged = order[0:partkilled]
    
    for x in partchanged:
        
        #parent selection
        parent1= -1*parent_selection_roulette(fit_pop)
        parent2= -1*parent_selection_roulette(fit_pop)
        
        for j in range(0,number_of_weights):
            
            prob1 = np.random.uniform(0,1)
            prob2 = np.random.uniform(0,1)
            if 0.5  <= prob1: #prob of changing the weight to the average of the two best individuals
                pop[x][j] = pop[order[parent1:]][0][j]*prob2 +  pop[order[parent2:]][0][j]*(1-prob2)
            
            prob3 = np.random.uniform(0,1)
            
            if mutation_prob <= prob3: #prob of changing the weight with an mutation 
                pop[x][j] = np.random.uniform(-1,1)
                
            
        fit_pop[x]=evaluate([pop[x]])

    return pop,fit_pop

# Generates random value in [-1,x) using a power function for bias
# Power>1 creates a bias towards x; Power<1 creates a bias towards -1
# size of power increases bias
def random_bias(x, power):
    return math.pow(random.random(), power) * (x + 1) - 1


#####################loading or creating a population#####################
# loads file with the best solution for testing
if run_mode =='test':

    bsol = np.loadtxt(experiment_name+'/best.txt')
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed','normal')
    evaluate([bsol])

    sys.exit(0)


# initializes population loading old solutions or generating new ones

if not os.path.exists(experiment_name+'/evoman_solstate'):

    print( '\nSTARTING A NEW EVOLUTION\n')

    pop = np.random.uniform(dom_l, dom_u, (npop, number_of_weights))
    fit_pop = evaluate(pop)
    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)
    ini_g = 0
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)

    print(fit_pop)
else:

    print( '\nCONTINUING WITH AN EVOLUTION\n')

    env.load_state()
    pop = env.solutions[0]
    fit_pop = env.solutions[1]

    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)

    # finds last generation number
    file_aux  = open(experiment_name+'/gen.txt','r')
    ini_g = int(file_aux.readline())
    file_aux.close()

# saves results for first pop
file_aux  = open(experiment_name+'/results.txt','a')
file_aux.write('\n\ngen best mean std')
print( '\n GENERATION '+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
file_aux.write('\n'+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
file_aux.close()
###################Evolution Algorithm########################

last_sol = fit_pop[best]
notimproved = 0
results = []
 
for i in range(ini_g+1, gens):
    
    
    pop, fit_pop = evolution(pop, fit_pop)  
    
    best = np.argmax(fit_pop) #best solution in generation
    fit_pop[best] = float(evaluate(np.array([pop[best] ]))[0]) # repeats best eval, for stability issues
    best_sol = fit_pop[best]

    if best_sol <= last_sol:
        notimproved += 1
    else:
        last_sol = best_sol
        notimproved = 0

    if notimproved >= 15:

        file_aux  = open(experiment_name+'/results.txt','a')
        file_aux.write('\ndoomsday')
        file_aux.close()

        pop, fit_pop = doomsday(pop,fit_pop)
        notimproved = 0

    best = np.argmax(fit_pop)
    std  =  np.std(fit_pop)
    mean = np.mean(fit_pop)

    results.append(fit_pop[best])
    plt.plot(results)
    plt.ylabel('Fitness')
    plt.xlabel('generation')
    plt.show()
    
    ###################save results#####################
    # saves results
    file_aux  = open(experiment_name+'/results.txt','a')
    print( '\n GENERATION '+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
    file_aux.write('\n'+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
    file_aux.close()

    # saves generation number
    file_aux  = open(experiment_name+'/gen.txt','w')
    file_aux.write(str(i))
    file_aux.close()

    # saves file with the best solution
    np.savetxt(experiment_name+'/best.txt',pop[best])
    # saves simulation state
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)
    env.save_state()

