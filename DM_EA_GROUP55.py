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

#number of....
n_hidden_neurons = 10

ini = time.time()  # sets time marker

enemies=[4]

#run type train or test 
run_mode = 'train' # train or test or trainten

if run_mode == 'test':
    experiment_name = 'test4.2'

#Var pop and NN
dom_u = 1 #upperbound NN value
dom_l = -1 #lowerbound NN value
npop = 5 #population size       #if changed check parent selection
gens = 3 #number of generations
mutation_prob = 0.20

def sim_environment(experiment_name, enemies):
    # initializes environment with ai player using random controller, playing against static enemy
    env = Environment(experiment_name=experiment_name,
                      enemies=enemies,  # Change enemies in top of file
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest")

    return env

######################function definitions########################3

#simulate with a specific NN
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))

def evolution(pop, fit_pop,i):
    
    partkilled = int(npop/2)  # a half of the population
    order = np.argsort(fit_pop)
    partchanged = order[0:partkilled]

    for x in partchanged:
        
        #parent selection (tournament)
        parent1 = tournament(fit_pop)
        parent2 = tournament(fit_pop)

        # crossover
        for j in range(0,number_of_weights):
            
            prob1 = np.random.uniform(0,1)

            if 0.5  <= prob1: #prob of changing the weight to the average of the two best individuals
                pop[x][j] = pop[parent1][j]
            else:
                pop[x][j] = pop[parent2][j]

            prob3 = np.random.uniform(0,1)
            
            mutation_prob = 1 - 0.9 * (i/gens) #variable mutation prob
            
            if mutation_prob <= prob3: #prob of changing the weight with an mutation 
                pop[x][j] = np.random.uniform(-1,1)
                
            
        fit_pop[x]=evaluate([pop[x]])

    return pop,fit_pop

# Choose best individual fitness-wise from 2 random candidates
def tournament(fit_pop):
    candidate_1, candidate_2 = np.random.choice(range(-npop, -1), 2)
    if fit_pop[candidate_1] > fit_pop[candidate_2]:
        return candidate_1
    else:
        return candidate_2

# Random weighted choice out of list integers
def random_choice(min, max, weights):
    parents = range(min, max+1)
    choice = np.random.choice(parents, p=weights)
    return choice


def kill_population(pop, fit_pop):
    
    partkilled = int(npop/2)  # a quarter of the population
    order = np.argsort(fit_pop)
    partchanged = order[0:partkilled]
    
    for x in partchanged:
        for j in range(0,number_of_weights):
        
            pop[x][j] = np.random.uniform(-1,1)
    
    return pop,fit_pop


#####################loading or creating a population#####################
# loads file with the best solution for testing
if run_mode =='test':
    env = sim_environment(experiment_name, enemies)
    bsol = np.loadtxt(experiment_name+'/best.txt')
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed','normal')
    evaluate([bsol])

    sys.exit(0)

if run_mode == 'trainten':
    total_runs = 10
else:
    total_runs = 1

# initializes population loading old solutions or generating new ones

for runs in range(total_runs):
    experiment_name = 'test{}.{}'.format(enemies[0], runs+1)
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # initializes environment with ai player using random controller, playing against static enemy
    env = sim_environment(experiment_name, enemies)

    # default environment fitness is assumed for experiment
    env.state_to_log()  # checks environment state

    # number of weights for multilayer with 10 hidden neurons
    number_of_weights = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

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

        pop, fit_pop = evolution(pop, fit_pop, i)

        best = np.argmax(fit_pop) #best solution in generation
        fit_pop[best] = float(evaluate(np.array([pop[best] ]))[0]) # repeats best eval, for stability issues
        best_sol = fit_pop[best]

        if best_sol <= last_sol:
            notimproved += 1
        else:
            last_sol = best_sol
            notimproved = 0

        if notimproved >= 5:

            file_aux  = open(experiment_name+'/results.txt','a')
            file_aux.write('\nReset the worst individuals to random.')
            file_aux.close()

            pop, fit_pop = kill_population(pop,fit_pop)
            notimproved = 0

        best = np.argmax(fit_pop)
        std  = np.std(fit_pop)
        mean = np.mean(fit_pop)

        results.append(fit_pop[best])

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

    if run_mode == 'train':
        plt.plot(results)
        plt.ylabel('Fitness')
        plt.xlabel('generation')
        plt.show()
