import random
import numpy as np
import array

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

from neural_network import NeuralNetwork

verbose = True

#targetFunc = lambda x: (x-0.5)**2
#targetFunc = lambda x: x**0.5
#targetFunc = lambda x: 0.4 if 0.3<x and x<0.7 else (0.1 if x<0.35 else 0.7)
targetFunc = lambda x: x**0.5 if x < 0.4 else (1-x+0.2)**2
#targetFunc = lambda x: x
#targetFunc = lambda x: 1-x
#targetFunc = lambda x: (1-x)**2
#targetFunc = lambda x : 0

psi = [1, 25, 12, 7, 15, 1]
sampling = 11
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

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    try:
        pop, log = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=ngen, 
                                       stats=stats, halloffame=hof, verbose=verbose)
    except KeyboardInterrupt as e:
        pass

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








