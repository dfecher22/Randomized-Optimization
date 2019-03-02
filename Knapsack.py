import mlrose
import numpy as np


weights=[11,15,3,4,5,6,10,13]
values=[5,7,1,2,1,3,4,11]

bweights=np.random.rand(30)
bvals=np.random.rand(30)



fitness=mlrose.Knapsack(weights,values,0.35)

init_state=np.zeros(len(weights))



problem = mlrose.DiscreteOpt(length = len(weights), fitness_fn = fitness,
                             maximize = True,max_val=2)

'''Simulated Annealing'''

schedule = mlrose.GeomDecay()

# Set random seed
np.random.seed(1)

# SA
opt_state, opt_fit = mlrose.simulated_annealing(problem, schedule = schedule,
                                                      max_attempts = 100000, max_iters = 1000000000000,
                                                      init_state = init_state)

print('The best state found is (SA): ',opt_state)
print('The fitness at the best state is (SA): ', opt_fit)

'''Random Hill Climb'''



# RHC
opt_state, opt_fit = mlrose.random_hill_climb(problem,
                                                      max_attempts = 5000, max_iters = 1000000000,restarts=50)

print('The best state found is (RHC): ',opt_state)
print('The fitness at the best state is (RHC): ', opt_fit)



# GA
opt_state, opt_fit = mlrose.genetic_alg(problem,max_attempts = 20, max_iters = 100)


print('The best state found is (GA): ',opt_state)
print('The fitness at the best state is (GA): ', opt_fit)


# MIMIC
opt_state, opt_fit = mlrose.mimic(problem,max_attempts = 20, max_iters = 1000)


print('The best state found is (mimic): ',opt_state)
print('The fitness at the best state is (mimic): ', opt_fit)