import numpy as np
import helpers

class NeuralNetwork():
    def __init__(self, weightMatrices, biases, psi, activationFunctions=None, costFunction=None):
        self.weightMatrices = weightMatrices
        self.biases = biases
        self.psi = psi
        self.depth = weightMatrices.shape[0]

        if self.depth == 1:
            self.forward = self._1_forward
        elif self.depth == 2:
            self.forward = self._2_forward
        elif self.depth == 3:
            self.forward = self._3_forward
        elif self.depth == 4:
            self.forward = self._4_forward
        else:
            self.forward = self._n_forward

        if activationFunctions == None:
            self.activationFunctions =  [helpers.elu] * (self.depth - 1) + [helpers.sigmoid]
            #self.activationFunctions = [helpers.linear] * (self.depth)
        else:
            self.activationFunctions = activationFunctions

        self.costFunction = costFunction
        if costFunction == None:
            self.costFunction = helpers.rmse

    @staticmethod
    def buildWeightsAndBiases(ind, psi):
        # number of biases
        n_biases = len(psi) - 1

        W = ind[:-n_biases]
        B = ind[-n_biases:]

        matricesElementCount = np.multiply(psi[:-1], psi[1:])
        splitIndices = np.cumsum(matricesElementCount)[:-1]
        M = np.split(W, splitIndices)
        weightMatrices = []
        for weights, layerIndex in zip(M, range(1, len(psi))):
            weightMatrices.append(weights.reshape(psi[layerIndex-1], psi[layerIndex]))
            
        weightMatrices = np.array(weightMatrices)

        return weightMatrices, B

    def _n_forward(self, x):
        layer = x
        for weightMatrix, activationFunction, bias in zip(self.weightMatrices,
                                                    self.activationFunctions, self.biases):
            layer = np.dot(layer, weightMatrix)
            layer = activationFunction(layer + bias)
        return layer

    def _1_forward(self, x):

        W = self.weightMatrices
        A = self.activationFunctions
        B = self.biases

        return A[0](np.dot(x, W[0]) + B[0])

    def _2_forward(self, x):

        W = self.weightMatrices
        A = self.activationFunctions
        B = self.biases

        return A[1](A[0](np.dot(x, W[0]) + B[0]).dot(W[1]) + B[1])

    def _3_forward(self, x):

        W = self.weightMatrices
        A = self.activationFunctions
        B = self.biases

        return A[2](A[1](A[0](np.dot(x, W[0]) + B[0]).dot(W[1]) + B[1]).dot(W[2]) + B[2])

    def _4_forward(self, x):

        W = self.weightMatrices
        A = self.activationFunctions
        B = self.biases

        return A[3](A[2](A[1](A[0](np.dot(x, W[0]) + B[0]).dot(W[1]) + B[1]).dot(W[2]) + B[2]).dot(W[3]) + B[3])
