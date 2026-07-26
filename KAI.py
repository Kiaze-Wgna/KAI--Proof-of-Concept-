from dataclasses import dataclass
import random
import math

# constants
bMin=-0.1
bMax=0.1

LINEAR = 0
LEAKY_RELU = 1
SIGMOID = 2
TANH = 3

@dataclass
class TrainingData:
    x_index: int
    inputs: list
    outputs: list

class Neuron:
    # 0: Linear 1: Leaky ReLU 2: Sigmoid 3: Tanh
    def __init__(self, ntype, wb=None, inN=None, outN=None):
        self.inputs = []
        self.preActivation = 0
        self.output = 0
        self.dead = False
        self.ntype = ntype
        if wb:
            self.weight = wb[0]
            self.bias = wb[1]
        else:
            if inN is None or outN is None:
                raise ValueError("Not enough information was given to create Neuron")
            if self.ntype in [LEAKY_RELU]: #He Initialization
                lim = math.sqrt(6 / inN)
            elif self.ntype in [LINEAR, SIGMOID, TANH]: #Xavier Initialization
                lim = math.sqrt(6 / (inN + outN))
            else:
                raise TypeError("Neuron Type Undefined")
            self.weight=[
                random.uniform(0 - lim, 0 + lim)
                for w in range(inN)
            ]
            self.bias=random.uniform(bMin,bMax)
        self.weightGradients = [0 for w in self.weight]
        self.biasGradient = 0
        self.batchSize = 0
        self.delta = 0
        self.weightVelocity = [0 for _ in self.weight]
        self.biasVelocity = 0
    def calculate(self, inputs):
        self.inputs = inputs
        self.preActivation = self.bias
        for i,w in zip(inputs, self.weight):
            self.preActivation += i*w
        if self.ntype == LINEAR:
            self.output = self.preActivation
        elif self.ntype == LEAKY_RELU:
            if self.preActivation > 0:
                self.output = self.preActivation
            else:
                self.output = 0.01 * self.preActivation
        elif self.ntype == SIGMOID:
            self.output = 1 / (1 + math.exp(-self.preActivation))
        elif self.ntype == TANH:
            self.output = math.tanh(self.preActivation)
    def activationDerivative(self):
        if self.ntype == LINEAR:
            return 1
        elif self.ntype == LEAKY_RELU:
            return 1 if self.preActivation > 0 else 0.01
        elif self.ntype == SIGMOID:
            return self.output * (1 - self.output)
        elif self.ntype == TANH:
            return 1 - self.output ** 2
    def returnWB(self):
        return [self.weight, self.bias]
    def updateWB(self, learningRate, momentum):
        if self.batchSize == 0:
            return

        for w in range(len(self.weight)):
            self.weightVelocity[w] = (
                (momentum * self.weightVelocity[w]) -
                learningRate * (
                    self.weightGradients[w] / self.batchSize
                )
            )
            self.weight[w] += self.weightVelocity[w]

        self.biasVelocity = (
            (momentum * self.biasVelocity) -
            learningRate * (
                self.biasGradient / self.batchSize
            )
        )
        self.bias += self.biasVelocity

        self.weightGradients = [0 for w in self.weight]
        self.biasGradient = 0
        self.batchSize = 0
            
class KAI:
    def __init__(self, inN, layer, weights=None, learningRate=0.01, momentum=0):
        self.initiateModel((inN, layer, weights, learningRate, momentum))
    def initiateModel(self, storedModel):
        self.inN, self.layer, weights, self.learningRate, self.momentum = storedModel
        self.model = []
        for l in range(len(self.layer)):
            self.model.append([])
            for n in range(self.layer[l][1]):
                if weights is None:
                    if l == 0:
                        inN = self.inN
                    else:
                        inN = self.layer[l - 1][1]
                    if l == len(self.layer) - 1:
                        outN = 1
                    else:
                        outN = self.layer[l + 1][1]
                    self.model[l].append(Neuron(self.layer[l][0], inN=inN, outN=outN))
                else:
                    self.model[l].append(Neuron(self.layer[l][0], wb=weights[l][n]))
    def calculate(self, inputs):
        if len(inputs) != self.inN:
            raise ValueError(f"Expected {self.inN} inputs, but received {len(inputs)} inputs.")
        self.inputs = inputs
        for l in range(len(self.layer)):
            self.outputs = []
            for n in range(self.layer[l][1]):
                self.model[l][n].calculate(self.inputs)
                self.outputs.append(self.model[l][n].output)
            self.inputs = self.outputs
    def distributeError(self, errorGradients):
        for l in reversed(range(len(self.layer))):
            for n in range(self.layer[l][1]):
                if l >= len(self.layer) - 1:
                    self.model[l][n].delta = errorGradients[n] * self.model[l][n].activationDerivative()
                else:
                    total = 0
                    for nextNeuron in self.model[l + 1]:
                        total+=nextNeuron.delta * nextNeuron.weight[n]

                    self.model[l][n].delta = total * self.model[l][n].activationDerivative()

                for i in range(len(self.model[l][n].weight)):
                    self.model[l][n].weightGradients[i] += self.model[l][n].delta * self.model[l][n].inputs[i]
                
                self.model[l][n].biasGradient += self.model[l][n].delta
                self.model[l][n].batchSize += 1
    def train(self, trainingDataSet, batchSize, lossFunction, gradientsFunction):
        errors=[]
        random.shuffle(trainingDataSet)
        currentSample = 0
        for trainingData in trainingDataSet:
            self.calculate(trainingData.inputs)
            errors.append(lossFunction(self.outputs, trainingData.outputs))
            self.distributeError(gradientsFunction(self.outputs, trainingData.outputs))
            currentSample += 1
            if currentSample == batchSize:
                currentSample = 0
                self.updateWB()
        if currentSample != 0:
            self.updateWB()
        return sum(errors) / len(errors)
    def storeModel(self):
        return (self.inN, self.layer,[
            [neuron.returnWB() for neuron in layer]
            for layer in self.model
        ], self.learningRate, self.momentum)
    def updateWB(self):
        for layer in self.model:
            for neuron in layer:
                neuron.updateWB(self.learningRate, self.momentum)