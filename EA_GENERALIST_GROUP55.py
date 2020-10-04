
# Author: Devin Mulder        			                                      
     				                                  


# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os


n_hidden_neurons = 10
enemy= [7,8]

<<<<<<< HEAD
experiment_name = 'multi_demo_4'
=======
experiment_name = 'multi_demo_2'
>>>>>>> parent of 31de02a... 30-9-2020
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes simulation in multi evolution mode, for multiple static enemies.
env = Environment(experiment_name=experiment_name,
                  enemies=[2,3],
                  multiplemode="yes",
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# default environment fitness is assumed for experiment

env.state_to_log() # checks environment state


####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

ini = time.time()  # sets time marker


# genetic algorithm params

run_mode = 'train' # train or test

# number of weights for multilayer with 10 hidden neurons.
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

dom_u = 1
dom_l = -1
<<<<<<< HEAD
npop = 5
gens = 3
mutation_prob = 0.4  # variable mutation prob
=======
npop = 10
gens = 20
mutation_prob = 0.2  # variable mutation prob
>>>>>>> parent of 31de02a... 30-9-2020

np.random.seed(69)

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))

def evolution(pop, fit_pop, npop):
    
    pop_new = np.random.uniform(dom_l, dom_u, (npop, n_vars))  
    fit_pop_new = []
    
    bestparents = int(npop/2) # a quarter of the population
    order = np.argsort(fit_pop)
    best_parents = order[bestparents:]   
    
    for x in pop_new:
        #parent selection (tournament)
        parent1 = tournament(best_parents, fit_pop)
        parent2 = tournament(best_parents, fit_pop)

        # crossover
        for j in range(0,n_vars):
            
            prob1 = np.random.uniform(0,1)

            pop_new[x][j] = pop[parent1][j] #+  0.5*pop[parent2][j]
            pop_new[x][j] = pop_new[x][j]*np.random.normal(0,0.5)
        
        
        fit_pop_new[x]=evaluate([pop_new[x]])
    
    
    fit_pop_new.append(fit_pop)
    pop_new.append(pop)
   
    order = np.argsort(fit_pop_new)
    best_individuals = order[npop:]
    pop = pop_new[best_individuals]
        
    return pop, fit_pop

# parent selection
def tournament(parent_range, fit_pop):
    candidate_1, candidate_2 = np.random.choice(parent_range, 2)
    if fit_pop[candidate_1] > fit_pop[candidate_2]:
        return candidate_1
    else:
        return candidate_2

#croos over
def cross_over():
    return

def mutation():
    return

# loads file with the best solution for testing
if run_mode =='test':

    bsol = np.loadtxt(experiment_name+'/best.txt')
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed','normal')
    evaluate([bsol])

    sys.exit(0)

# initializes population loading old solutions or generating new ones
if not os.path.exists(experiment_name+'/evoman_solstate'):

    print( '\nNEW EVOLUTION\n')

    pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
    fit_pop = evaluate(pop)
    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)
    ini_g = 0
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)

else:

    print( '\nCONTINUING EVOLUTION\n')

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


# evolution

last_sol = fit_pop[best]
notimproved = 0

for i in range(ini_g+1, gens):

    pop, fit_pop = evolution(pop, fit_pop, npop)

    best = np.argmax(fit_pop)  # best solution in generation
    fit_pop[best] = float(evaluate(np.array([pop[best]]))[0])  # repeats best eval, for stability issues
    best_sol = fit_pop[best]

    best = np.argmax(fit_pop)
    std  =  np.std(fit_pop)
    mean = np.mean(fit_pop)


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

fim = time.time() # prints total execution time for experiment
print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')


file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
file.close()


env.state_to_log() # checks environment state
