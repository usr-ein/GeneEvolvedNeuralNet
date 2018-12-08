import random
import numpy as np

from deap import base
from deap import creator
from deap import tools

from neural_network import NeuralNetwork

from grapher import Grapher


grapher = Grapher("NN fitness over gen", "Generation Nº", "Avg fitness", color='red')
graphing = True
verbose = True

#targetFunc = lambda x: (x-0.5)**2
#targetFunc = lambda x: x**0.5
#targetFunc = lambda x: x**0.2345
#targetFunc = lambda x: 0.4 if 0.3<x and x<0.7 else (0.1 if x<0.35 else 0.7)
targetFunc = lambda x: x**0.5 if x < 0.4 else (1-x+0.2)**2
#targetFunc = lambda x: x
#targetFunc = lambda x: 1-x
#targetFunc = lambda x: (1-x)**2

psi = [1, 25, 12, 7, 15, 1]
sampling = 7
ind_size = (np.multiply(psi[:-1], psi[1:]).sum()) + (len(psi) - 1)
ngen = 1000000
npop = 400
hof_size = 5

# CXPB  is the probability with which two individuals are crossed
# MUTPB is the probability for mutating an individual
CXPB, MUTPB = 0.6, 0.2

random.seed(0)
np.random.seed(0)

# Name of new class, base class, kwargs* that become attributes of new class
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# Attribute generator
toolbox.register("attr_float", random.uniform, -1, 1)
# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, ind_size)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalNet(ind, randEval=False):
    """ When randEval is True, inputs will be sampled at random points,
        when False, inputs will be at regular intervals
    """
    W, B = NeuralNetwork.buildWeightsAndBiases(ind, psi)
    nn = NeuralNetwork(weightMatrices=W, biases=B, psi=psi)
    fit = 0
    linspace = np.linspace(0,1,sampling)
    for i in range(sampling):
        if randEval:
            inputs = np.random.rand(psi[0])
        else:
            inputs = linspace[i]
        output = nn.forward(inputs)
        target = targetFunc(inputs)
        cost = nn.costFunction(output, target)
        fit += cost

    return fit,

toolbox.register("evaluate", evalNet)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, indpb=0.05, mu=0, sigma=0.5)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    pop = toolbox.population(n=npop)
    hof = tools.HallOfFame(hof_size)

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    gen = 0
    try:
        # Begin the evolution
        while gen < ngen:
            # A new generation
            gen += 1
            # Select the next generation individuals
            offspring = toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))
            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            pop[:] = offspring

            hof.update(pop)

            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in pop]
            
            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x*x for x in fits)
            std = abs(sum2 / length - mean**2)**0.5
            
            if verbose: print("G {:4d} | _f {:2E} | ^f {:2E} | ≈f {:2E} | σ {:2E}".format(gen, min(fits), max(fit), mean, std))
            if not verbose: print(" G {}".format(gen), end='\r')
            if graphing: grapher.newPoint(gen, mean)

    except KeyboardInterrupt as e:
        pass

    print(" "*10)
    showResult(pop, hof)


def showResult(pop, hof):
    weights_biases = [NeuralNetwork.buildWeightsAndBiases(ind, psi) for ind in hof]

    print("Best Weights:")
    print(weights_biases[0][0])
    print("Best Biases:")
    print(weights_biases[0][1])

    nets = [NeuralNetwork(weightMatrices=W, biases=B, psi=psi) for W, B in weights_biases]

    funcs = [nn.forward for nn in nets] + [targetFunc]
    grapher.plotFuncs(funcs)

if __name__ == '__main__':
    main()
