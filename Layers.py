import numpy as np
from functools import reduce

# Dense layer
class DenseLayer:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases.
        # The function random.randn generate data randomly sampled from a Gaussian distribution with a mean of 0.
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.neurons = n_neurons
        # Initializing all biases as zero.
        self.biases = np.zeros((1, n_neurons))
        self.output = None
        self.inputs = None
        self.zs = None
        self.nablaW = None
        self.nablaB = None
        self.decay = 1e-6
        self.curLearningRate = 0
        self.iterations = 0
        self.weightMomentums = np.zeros(self.weights.shape)
        self.biasMomentums = np.zeros(self.biases.shape)
        self.momentum = .95
        self.cacheWeights = np.zeros(self.weights.shape)
        self.cacheBiases = np.zeros(self.biases.shape)

    def preUpdateParams(self, learningRate):
        if self.curLearningRate == 0:
            self.curLearningRate = learningRate

        self.curLearningRate = learningRate*(1./(1. + self.decay * self.iterations))
        self.iterations+=1

    def adagradUpdate(self, miniBatchSize):
        self.cacheWeights += self.nablaW**2
        self.cacheBiases += self.nablaB**2
        self.weights += - (self.curLearningRate / miniBatchSize) * self.nablaW / (np.sqrt(self.cacheWeights) + 1e-7)
        self.biases += - (self.curLearningRate / miniBatchSize) * self.nablaB / (np.sqrt(self.cacheBiases) + 1e-7)

    def updateParams(self, miniBatchSize):
        return self.adagradUpdate(miniBatchSize)
        # weightsUpdate = self.momentum * self.weightMomentums - (self.curLearningRate / miniBatchSize) * self.nablaW
        # biasUpdate = self.momentum * self.biasMomentums - (self.curLearningRate / miniBatchSize) * self.nablaB
        #
        # self.weights += weightsUpdate
        # self.biases += biasUpdate
        #
        # self.weightMomentums = weightsUpdate
        # self.biasMomentums = biasUpdate
        #self.weights += - (self.curLearningRate / miniBatchSize) * self.nablaW
        #self.biases += - (self.curLearningRate / miniBatchSize) * self.nablaB

    # Forward pass
    def forward(self, inputs):
        self.inputs = np.array(inputs)
        # Calculate output values from inputs, weights and biases
        # print("input and weights: \n", inputs[:5],"\n", self.weights[:5], "\n\n")
        self.zs = np.dot(self.inputs, self.weights) + self.biases
        # Applying the rectified linear activation function
        return self.activationFunction()

    def activationFunction(self):
        pass



class HiddenLayer(DenseLayer):
    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        super().__init__(n_inputs, n_neurons)

    # Forward pass
    def forward(self, inputs):
        return super().forward(inputs)

    # Applying ReLU (rectified linear activation function).
    def activationFunction(self):
        self.output = np.maximum(0, self.zs)
        return self.output

    def backward(self, nextLayerGradient, learningRate):
        self.preUpdateParams(learningRate)
        miniBatchSize = len(nextLayerGradient)
        # accumulators
        self.nablaB = np.zeros(self.biases.shape)
        self.nablaW = np.zeros(self.weights.shape)

        dRelu = np.zeros(self.zs.shape)
        dRelu[self.zs > 0] = 1

        dReluNextLayerGradient = nextLayerGradient * dRelu
        dLoss_dx = np.dot(dReluNextLayerGradient, self.weights.T)

        self.nablaW = np.dot(self.inputs.T, dReluNextLayerGradient)
        self.nablaB = np.sum(dReluNextLayerGradient, axis=0, keepdims=True)

        self.updateParams(miniBatchSize)

        return dLoss_dx


class OutputLayer(DenseLayer):
    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        super().__init__(n_inputs, n_neurons)

    # Forward pass
    def forward(self, inputs):
        return super().forward(inputs)

    # Applying SoftMax activation function. Basically : P_i = e^o_i/sum(e^o_k) for all k.
    def activationFunction(self):
        # SoftMax fucntion modified to prevent overflow and large numbers
        #probabilities = []
        #try:
        expValues = [np.exp(outputSample - np.max(outputSample)) for outputSample in self.zs]
        probabilities = [sample/np.sum(sample) for sample in expValues]
        # except:
        #     print(self.zs)
        # Better format:
        self.output = np.array(probabilities)
        return self.output

    def backward(self, trueOutputClasses, learningRate):
        self.preUpdateParams(learningRate)
        miniBatchSize = len(trueOutputClasses)
        # accumulators
        self.nablaW = np.zeros(self.weights.shape)
        self.nablaB = np.zeros(self.biases.shape)

        oneHotEncoded = np.zeros(self.output.shape)
        for row,k in enumerate(trueOutputClasses):
            oneHotEncoded[row][k] = 1.

        dLoss_dz = (self.output - oneHotEncoded)
        dLoss_dx = np.dot(dLoss_dz, self.weights.T)  # this will output sample x inputs.

        self.nablaW = np.dot(self.inputs.T, dLoss_dz)
        self.nablaB = np.sum(dLoss_dz, axis=0, keepdims=True)

        self.updateParams(miniBatchSize)

        return dLoss_dx




