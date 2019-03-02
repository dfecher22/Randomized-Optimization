import mlrose
import numpy as np



fitness=mlrose.FourPeaks(0.2)

init_state=np.zeros(50)

problem = mlrose.DiscreteOpt(length = init_state.shape[0], fitness_fn = fitness,
                             maximize = True,max_val=2)

'''Simulated Annealing'''

schedule = mlrose.GeomDecay()

# Set random seed
np.random.seed(1)

# SA
'''opt_state, opt_fit = mlrose.simulated_annealing(problem, schedule = schedule,
                                                      max_attempts = 1000, max_iters = 100000000,
                                                      init_state = init_state)

print('The best state found is (SA): ',opt_state)
print('The fitness at the best state is (SA): ', opt_fit)




# Hill Climb
opt_state, opt_fit = mlrose.random_hill_climb(problem,
                                                      max_attempts = 5000, max_iters = 1000000000)

print('The best state found is (RHC): ',opt_state)
print('The fitness at the best state is (RHC): ', opt_fit)'''

'''Genetic Alg'''



# GA
'''opt_state, opt_fit = mlrose.genetic_alg(problem,mutation_prob=0.005,pop_size=10000,max_attempts = 50, max_iters = 2500)


print('The best state found is (GA): ',opt_state)
print('The fitness at the best state is (GA): ', opt_fit)'''



# Mimic
opt_state, opt_fit = mlrose.mimic(problem,pop_size=1000,max_attempts = 100, max_iters = 300)


print('The best state found is (mimic): ',opt_state)
print('The fitness at the best state is (mimic): ', opt_fit)