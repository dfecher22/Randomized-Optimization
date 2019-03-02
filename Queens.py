import mlrose
import numpy as np

'''Simulated Annealing'''

fitness=mlrose.Queens()

schedule = mlrose.ExpDecay()

problem = mlrose.DiscreteOpt(length = 8, fitness_fn = fitness,
                             maximize = False, max_val = 8)

# Define initial state
init_state = np.array([1, 1, 1, 1, 1, 1, 1, 5])

# Set random seed
np.random.seed(1)

# Solve problem using simulated annealing
opt_state, opt_fit = mlrose.simulated_annealing(problem, schedule = schedule,
                                                      max_attempts = 100, max_iters = 10000,
                                                      init_state = init_state)

print('The best state found is (SA): ',opt_state)
print('The fitness at the best state is (SA): ', opt_fit)

#RHC

opt_state, opt_fit = mlrose.random_hill_climb(problem,
                                                      max_attempts = 200, max_iters = 1000,restarts=10,
                                                      init_state = init_state)

print('The best state found is (RHC): ',opt_state)
print('The fitness at the best state is (RHC): ', opt_fit)

'''Genetic Alg'''



#GA

opt_state, opt_fit = mlrose.genetic_alg(problem,max_attempts = 100,pop_size=15000,mutation_prob=0.05, max_iters = 1000)


print('The best state found is (GA): ',opt_state)
print('The fitness at the best state is (GA): ', opt_fit)




# MIMIC
opt_state, opt_fit = mlrose.mimic(problem,max_attempts = 20,pop_size=2500, max_iters = 1500)


print('The best state found is (mimic): ',opt_state)
print('The fitness at the best state is (mimic): ', opt_fit)
