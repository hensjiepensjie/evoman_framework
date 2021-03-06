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


def new_evolution(pop, fit_pop, npop, gen_pop, gen, total_gens, env):
    offspring = []
    n_mutants = int(npop / 2) # Half of population
    # Crossover
    # Integer list to make random pairs of parents from full population (If using this: uncomment line 43-47 and comment line 50-54)
    #int_list = np.arange(npop)
    #mating_parents = np.random.choice(int_list, size=(int(npop / 2), 2), replace=False)
    #for pair in mating_parents:
    #    child = cross_over(pair, pop)
    #    offspring.append(child)

    # Choose pairs with tournament (If using this: comment line 43-47 and uncomment line 50-54)
    for i in range(n_mutants):
        parent_1 = tournament(len(pop), fit_pop)
        parent_2 = tournament(len(pop), fit_pop)
        child = cross_over([parent_1, parent_2], pop)
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


# parent selection
def tournament(parent_range, fit_pop):
    candidate_1, candidate_2 = np.random.choice(parent_range, 2, replace=False)
    if fit_pop[candidate_1] > fit_pop[candidate_2]:
        return candidate_1
    else:
        return candidate_2


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
    #np.random.seed(69)
    n_hidden_neurons = args.n_neurons

    ini = time.time()  # sets time marker

    enemies = args.enemies

    # run type train or test
    run_mode = args.run_mode  # train or test or trainten

    # run type train or test
    if run_mode == 'test' or run_mode == 'test_final':
        experiment_name = args.experiment_name  # train or test or trainten

    dom_u = 1
    dom_l = -1
    npop = args.npop  # population size
    gens = args.gens  # number of generations

    # loads file with the best solution for testing
    if run_mode == 'test':
        env = sim_environment(experiment_name, enemies, n_hidden_neurons)
        bsol = np.loadtxt(experiment_name + '/best.txt')
        print('\n RUNNING SAVED BEST SOLUTION \n')
        env.update_parameter('speed', 'normal')
        evaluate([bsol], env)

        sys.exit(0)

    # loads file with the best solution for testing
    if run_mode == 'test_final':
        env = sim_environment(experiment_name, [1,2,3,4,5,6,7,8], n_hidden_neurons)
        bsol = np.loadtxt(experiment_name + '/final_sol.txt')
        print('\n RUNNING SAVED BEST FINAL SOLUTION \n')
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
            ini_sol = 0
            best_fit_sol = 0
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

            if os.path.exists(experiment_name + '/gen_sol.txt'):

                # finds last solution line
                file_aux = open(experiment_name + '/gen_sol.txt', 'r')
                ini_sol = int(file_aux.readline())
                file_aux.close()

            else:
                ini_sol = 0

            if os.path.exists(experiment_name + '/best_fit_sol.txt'):
                # finds last best final fitness
                file_aux = open(experiment_name + '/best_fit_sol.txt', 'r')
                best_fit_sol = float(file_aux.readline())
                file_aux.close()

            else:
                best_fit_sol = 0
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
        file_aux.write(str(ini_g) + ',')
        np.savetxt(file_aux, pop[best], newline=',')
        file_aux.close()

        last_sol = fit_pop[best]
        notimproved = 0
        results = []

        for i in range(ini_g+1, gens):
            # Save old values
            old_best = best
            old_best_sol = fit_pop[old_best]

            # Delete worst half and generate random individuals every 20 generations
            if i % 20 == 0:
                order = np.argsort(fit_pop)
                new_pop = np.random.uniform(dom_l, dom_u, (int(npop/2), number_of_weights))
                pop[order[:int(npop/2)]] = new_pop

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

    final_sol = full_training_test(experiment_name, number_of_weights, n_hidden_neurons, ini_sol, best_fit_sol)
    # saves file with the best solution
    np.savetxt(experiment_name + '/final_sol.txt', final_sol)


def full_training_test(experiment_name, number_of_weights, n_hidden_neurons, ini_sol, best_fit_sol):
    print('\n Running best solutions against all 8 enemies')
    env_8 = sim_environment(experiment_name, [1,2,3,4,5,6,7,8], n_hidden_neurons)
    bsols = np.loadtxt(experiment_name + '/bestsols.txt', delimiter = ',', usecols=range(1, number_of_weights+1))

    best_final_solution = []

    # If only one solution found check and return
    if len(bsols.shape) == 1:
        best_fit_sol = evaluate([bsols], env_8)[0]
        best_final_solution = bsols
        # saves solution number
        file_aux = open(experiment_name + '/gen_sol.txt', 'w')
        file_aux.write(str(1))
        file_aux.close()

    else:
        for sol in range(ini_sol, len(bsols)):
            fit_sol = evaluate([bsols[sol]], env_8)
            if fit_sol > best_fit_sol:
                best_final_solution = bsols[sol]
                best_fit_sol = fit_sol[0]

            # Save current line in solution for continuing later
            file_aux = open(experiment_name + '/gen_sol.txt', 'w')
            file_aux.write(str(sol))
            file_aux.close()

    print("Best fitness value against 8 enemies is %g" % best_fit_sol)

    file_aux = open(experiment_name + '/best_fit_sol.txt', 'w')
    file_aux.write(str(best_fit_sol))
    file_aux.close()

    # If no best_final_solution is found after continuing, quit
    if len(best_final_solution) == 0:
        print('no better solution found than previous')
        sys.exit(0)

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
    parser.add_argument('-experiment_name', '--experiment_name', default='multi_demo_1357')
    args = parser.parse_args(sys.argv[1:])

    run_simulation(args)