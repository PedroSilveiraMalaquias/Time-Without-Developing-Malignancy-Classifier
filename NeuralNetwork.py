import random

from Layers import *
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:

    def __init__(self, neuronsPerLayer: list = []):
        self.targetOutput = None
        self.curInputOrder = None
        self.features = neuronsPerLayer[0]
        self.outputClasses = neuronsPerLayer[-1]
        self.nLayers = len(neuronsPerLayer) - 1
        self.neuronsPerLayer = neuronsPerLayer[1:]
        self.layers = []
        self.trainingCycles = 8000
        self.miniBatchSize = 20
        self.learningRate = 1.
        self.epoch = 0
        self.higherAccuracy = 0

        nInput = self.features
        for i in range(self.nLayers - 1):
            self.layers.append(HiddenLayer(nInput, self.neuronsPerLayer[i]))
            nInput = self.neuronsPerLayer[i]

        self.layers.append(OutputLayer(nInput, self.outputClasses))

    def evaluate(self, testData, accuracies = [], meanLosses = []):
        x = [sample[0] for sample in testData]
        y = [sample[1] for sample in testData]

        curInput = x
        for layer in self.layers:
            curInput = layer.forward(curInput)

        probabilityPerClass = self.layers[-1].output

        losses = self.crossEntropyLoss(probabilityPerClass, y)
        meanLoss = np.mean(losses)
        predictions = [np.argmax(sampleOutput) for sampleOutput in probabilityPerClass]
        accuracy = np.mean([int(prediction == target) for prediction, target
                            in zip(predictions, y)])

        self.higherAccuracy = max(self.higherAccuracy, accuracy)
        accuracies.append(accuracy)
        meanLosses.append(meanLoss)
        print("Epoch: ", self.epoch, " Mean Loss: ", meanLoss, " Accuracy: ", accuracy,
              " Learning Rate: ", self.layers[-1].curLearningRate)

    def train(self, trainingData, testData):

        accuracies = []
        meanLosses = []
        for cycle in range(self.trainingCycles):
            self.epoch = cycle
            random.shuffle(trainingData)
            miniBatches = [
                 trainingData[k:k + self.miniBatchSize]  # taking partitions of size miniBatchSize
                 for k in range(0, len(trainingData), self.miniBatchSize)]

            self.targetOutput = []
            self.curInputOrder = []
            for miniBatch in miniBatches:
                x = [sample[0] for sample in miniBatch]
                y = [sample[1] for sample in miniBatch]

                curInput = x
                for layer in self.layers:
                    curInput = layer.forward(curInput)

                curGradient = self.layers[-1].backward(y, self.learningRate)
                for i in range(self.nLayers - 2, -1, -1):
                    curGradient = self.layers[i].backward(curGradient, self.learningRate)

                self.targetOutput.extend(y)
                self.curInputOrder.extend(x)

            self.evaluate(testData, accuracies, meanLosses)

        print("Highest accuracy: ", self.higherAccuracy," Training Samples: ",len(trainingData),
              " Test Samples: ",len(testData))
        axisX = [i for i in range(self.trainingCycles)]
        plt.plot(axisX, meanLosses)
        plt.plot(axisX, accuracies, color='red')
        plt.show()

    # Categorical Cross Entropy Loss Function formula: loss_k = sum(-alog(â)) for each sample k.
    def crossEntropyLoss(self, probabilityPerClass, targetOutput):

        targetClassLoss = np.array([distribution[targetClass] for targetClass, distribution
                                    in zip(targetOutput, probabilityPerClass)])

        # Defining a lower and an upper bound to prevent log(0) and mean shifting
        clippedTargetClassLoss = np.clip(targetClassLoss, 1e-7, 1 - 1e-7)
        loss = -np.log(clippedTargetClassLoss)
        return loss
