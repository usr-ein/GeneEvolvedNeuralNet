# Genetically Evolved Neural Network

![Results from approximating an arbitrary 1D function](https://github.com/sam1902/GeneEvolvedNeuralNet/raw/master/Result.png)

## What is this program ? Why is it there ?
This program is a small project designed to get used to DEAP, a genetic algorithm Python framework that greatly eases the developping process by providing a structure to populations, individuals and genetic operators handling.

This program also follows my multiple failed attempts at implementing the [CoSyNE algorithm](https://github.com/sam1902/CoSyNEPy) by hand in vanilla NumPy. In the future, I'm planning on modifying this program to incorporate the changes specific to the CoSyNE algorithm.


## What does it do ?
But as for now, I have this program that can - with a simple genetic algorithm - evolve the weights and biases necessaries to build a network that will mimic the defined `targetFunc` for x in [0, 1] and y in [0, 1]. These intervals can easily be extended to include negative values as well but I'm not interested in doing that for now.

It can graph in **live the evolution of the fitness**, which is the same as the cost here, as we're trying to minimise it. Furthermore, when Ctrl+C is pressed or that the maximum numbre of generations `n_gen` is reached, it displays the best individual's weights and biases as well as **the graphs of the best `hof_size` functions (by default 5) along with the `targetFunc`'s graph.**


So this can be used for educational purposes.
