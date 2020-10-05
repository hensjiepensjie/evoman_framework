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


# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(x, env):
    return np.array(list(map(lambda y: simulation(env,y), x)))

def evolution(pop, fit_pop, npop, number_of_weights, env):
    
    partkilled = int(3*npop/4)  # a quarter of the population
    bestparents = int(npop/4) # a half of the population
    order = np.argsort(fit_pop)
    partchanged = order[0:partkilled]
    best_parents = order[bestparents:]
         
    for x in partchanged:
        #parent selection (tournament)
        parent1 = tournament(best_parents, fit_pop)
        parent2 = tournament(best_parents, fit_pop)

        # crossover
        for j in range(0, number_of_weights):
            
            prob1 = np.random.uniform(0,1)

            if 0.5  <= prob1: #prob of changing the weight to the average of the two best individuals
                pop[x][j] = pop[parent1][j]
            else:
                pop[x][j] = pop[parent2][j]

        fit_pop[x] = evaluate([pop[x]], env)
        
    return pop, fit_pop

# parent selection
def tournament(parent_range, fit_pop):
    candidate_1, candidate_2 = np.random.choice(parent_range, 2)
    if fit_pop[candidate_1] > fit_pop[candidate_2]:
        return candidate_1
    else:
        return candidate_2


def new_evolution(pop, fit_pop, npop, gen_pop, gen, total_gens, env):
    offspring = []
    n_mutants = int(npop / 2) # Half of population
    # Crossover
    # Integer list to make random pairs of parents from full population
    int_list = np.arange(npop)
    mating_parents = np.random.choice(int_list, size=(int(npop / 2), 2), replace=False)
    for pair in mating_parents:
        child = cross_over(pair, pop)
        offspring.append(child)

    # Mutation over parents and children sampled
    ints_to_mutate = np.random.choice(npop+len(offspring), n_mutants, replace=False)

    for integer in ints_to_mutate:
        # If integer smaller than npop, than mutate parent. Otherwise mutate child.
        if integer < npop:
            mutant = pop[integer]
            sigma = (total_gens - gen) / total_gens

        else:
            mutant = offspring[integer-npop]
            sigma = 0.5

        child = mutation(mutant, sigma)
        offspring.append(child)

    fit_offspring = evaluate(offspring, env)

    # Best npop of parents and children stay alive
    # Copy lists and concatenate
    current_gen = np.array([gen]*npop)
    temp_pop = np.concatenate((pop, offspring))
    temp_fit = np.concatenate((fit_pop, fit_offspring))
    temp_gen_pop = np.concatenate((gen_pop, current_gen))

    surviving_order = np.argsort(temp_fit)
    # Best npop become new population
    pop = temp_pop[surviving_order][-npop:]
    fit_pop = temp_fit[surviving_order][-npop:]
    gen_pop = temp_gen_pop[surviving_order][-npop:]

    return pop, fit_pop, gen_pop


# convex combination of two parents
def cross_over(pair, pop):
    random_1 = np.random.uniform()
    random_2 = np.random.uniform()

    child = pop[pair[0]]*random_1 + pop[pair[1]]*random_2

    return child


# Add value from gaussian distribution with 0 mean and sigma standard deviation to weight
def mutation(mutant, sigma):

    # Loop over each weight
    for i in range(len(mutant)):
        mutant[i] = mutant[i] + np.random.normal(loc=0.0, scale=sigma)

    return mutant


def run_simulation(args):
    np.random.seed(69)
    n_hidden_neurons = args.n_neurons

    ini = time.time()  # sets time marker

    enemies = args.enemies

    # run type train or test
    run_mode = args.run_mode  # train or test or trainten

    # run type train or test
    if run_mode == 'test':
        experiment_name = args.experiment_name  # train or test or trainten

    dom_u = 1
    dom_l = -1
    npop = args.npop  # population size
    gens = args.gens  # number of generations

    mutation_prob = 0.2  # variable mutation prob

    # loads file with the best solution for testing
    if run_mode == 'test':
        env = sim_environment(experiment_name, enemies, n_hidden_neurons)
        bsol = np.loadtxt(experiment_name + '/best.txt')
        print('\n RUNNING SAVED BEST SOLUTION \n')
        env.update_parameter('speed', 'normal')
        evaluate([bsol], env)

        sys.exit(0)

    if run_mode == 'trainten':
        total_runs = 10
    else:
        total_runs = 1

    for runs in range(total_runs):
        if run_mode == 'trainten':
            experiment_name = 'multi_demo_{}.{}'.format(enemies, runs + 1)
            if not os.path.exists(experiment_name):
                os.makedirs(experiment_name)
        elif run_mode == 'train':
            experiment_name = args.experiment_name
            if not os.path.exists(experiment_name):
                os.makedirs(experiment_name)

        # initializes environment with ai player using random controller, playing against static enemy
        env = sim_environment(experiment_name, enemies, n_hidden_neurons)

        # default environment fitness is assumed for experiment

        env.state_to_log()  # checks environment state

        ####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

        # number of weights for multilayer with 10 hidden neurons.
        number_of_weights = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

        # evolution

        # initializes population loading old solutions or generating new ones
        if not os.path.exists(experiment_name + '/evoman_solstate'):

            print('\nNEW EVOLUTION\n')

            pop = np.random.uniform(dom_l, dom_u, (npop, number_of_weights))
            gen_pop = np.zeros(npop, dtype='int') # List of numbers at which generation someone was born
            fit_pop = evaluate(pop, env)
            best = np.argmax(fit_pop)
            mean = np.mean(fit_pop)
            std = np.std(fit_pop)
            ini_g = 0
            solutions = [pop, fit_pop, gen_pop]
            env.update_solutions(solutions)

        else:

            print('\nCONTINUING EVOLUTION\n')

            env.load_state()
            pop = env.solutions[0]
            fit_pop = env.solutions[1]
            gen_pop = env.solutions[2]

            best = np.argmax(fit_pop)
            mean = np.mean(fit_pop)
            std = np.std(fit_pop)

            # finds last generation number
            file_aux = open(experiment_name + '/gen.txt', 'r')
            ini_g = int(file_aux.readline())
            file_aux.close()

        # saves results for first pop
        file_aux = open(experiment_name + '/results.txt', 'a')
        file_aux.write('\n\ngen best mean std best_sol')
        print(
            '\n GENERATION ' + str(ini_g) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(
                round(std, 6)))
        file_aux.write('\n' + str(ini_g) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(
            round(std, 6)))
        file_aux.close()

        # saves results for first pop
        file_aux = open(experiment_name + '/bestsols.txt', 'a')
        file_aux.write(str(0) + ',')
        np.savetxt(file_aux, pop[best], newline=',')
        file_aux.close()

        last_sol = fit_pop[best]
        notimproved = 0
        results = []

        for i in range(ini_g+1, gens):
            # Save old values
            old_best = best
            old_best_sol = fit_pop[old_best]

            pop, fit_pop, gen_pop = new_evolution(pop, fit_pop, npop, gen_pop, i, gens, env)
            best = np.argmax(fit_pop)  # best solution in generation

            fit_pop[best] = float(evaluate(np.array([pop[best]]), env)[0])  # repeats best eval, for stability issues
            best_sol = fit_pop[best]

            best = np.argmax(fit_pop)
            std  = np.std(fit_pop)
            mean = np.mean(fit_pop)

            results.append(fit_pop[best])

            # saves results
            file_aux  = open(experiment_name+'/results.txt','a')
            print( '\n GENERATION '+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
            file_aux.write('\n'+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
            file_aux.close()

            # Saves best solution if different from previous best
            if old_best_sol != best_sol:
                file_aux = open(experiment_name + '/bestsols.txt', 'a')
                file_aux.write("\n")
                file_aux.write(str(i) + ',')
                np.savetxt(file_aux, pop[best], newline=',')
                file_aux.close()

            # saves generation number
            file_aux  = open(experiment_name+'/gen.txt','w')
            file_aux.write(str(i))
            file_aux.close()

            # saves file with the best solution
            np.savetxt(experiment_name+'/best.txt', pop[best])

            # saves simulation state
            solutions = [pop, fit_pop, gen_pop]
            env.update_solutions(solutions)
            env.save_state()

        fim = time.time() # prints total execution time for experiment
        print('\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')

        file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
        file.close()

    final_sol = full_training_test(experiment_name, number_of_weights, n_hidden_neurons)
    # saves file with the best solution
    np.savetxt(experiment_name + '/final_sol.txt', final_sol)


def full_training_test(experiment_name, number_of_weights, n_hidden_neurons):
    print('\n Running best solutions against all 8 enemies')
    env_8 = sim_environment(experiment_name, [1,2,3,4,5,6,7,8], n_hidden_neurons)
    bsols = np.loadtxt(experiment_name + '/bestsols.txt', delimiter = ',', usecols=range(1, number_of_weights+1))

    best_fit_sol = 0
    best_final_solution = []

    # If only one solution found check and return
    if len(bsols.shape) == 1:
        best_fit_sol = evaluate([bsols], env_8)
        best_final_solution = bsols

    else:
        for sol in range(len(bsols)):
            fit_sol = evaluate([bsols[sol]], env_8)
            if fit_sol > best_fit_sol:
                best_final_solution = bsols[sol]
                best_fit_sol = fit_sol

    print("Best fitness value against 8 enemies is %g" % best_fit_sol)

    return best_final_solution


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('-n_neurons', '--n_neurons', default=10, type=int)
    parser.add_argument('-enemies', '--enemies', nargs='+', default=[2, 3])
    parser.add_argument('-npop', '--npop', default=40, type=int),
    parser.add_argument('-gens', '--gens', type=int, default=20)
    parser.add_argument('-run_mode', '--run_mode', default='train')
    parser.add_argument('-experiment_name', '--experiment_name', default='multi_demo')
    args = parser.parse_args(sys.argv[1:])

    run_simulation(args)
    #full_training_test('multi_demo', 265, 10)