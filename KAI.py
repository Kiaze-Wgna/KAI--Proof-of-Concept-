import random

# constants
wMin=-1
wMax=1
bMin=-0.1
bMax=0.1

def average(lis):
    return sum(lis) / len(lis)

class Neuron:
    def __init__(self, wb, ntype):
        self.inputs = []
        self.ogOutput = 0
        self.output = 0
        self.dead = False
        self.ntype = ntype
        if type(wb) is list:
            self.weight = wb[0]
            self.bias = wb[1]
        else:
            self.weight=[
                random.uniform(wMin,wMax)
                for w in range(wb)
            ]
            self.bias=random.uniform(bMin,bMax)
        self.weightGradients = [0 for w in self.weight]
        self.biasGradient = 0
        self.batchSize = 0
        self.delta = 0
    def calculate(self, inputs):
        self.inputs = inputs

        self.ogOutput = self.bias
        for i,w in zip(inputs, self.weight):
            self.ogOutput += i*w
        if self.ntype == 1:
            if self.ogOutput > 0:
                self.output = self.ogOutput
            else:
                self.output = 0.01 * self.ogOutput
        else:
            self.output = self.ogOutput
    def activationDerivative(self):
        if self.ntype == 1:
            return 1 if self.ogOutput > 0 else 0.01
        return 1
    def returnWB(self):
        return [self.weight, self.bias]
    def updateWB(self, learningRate):
        if self.batchSize == 0:
            return

        for w in range(len(self.weight)):
            self.weight[w] -= learningRate * (
                self.weightGradients[w] / self.batchSize
            )

        self.bias -= learningRate * (
            self.biasGradient / self.batchSize
        )

        self.weightGradients = [0 for w in self.weight]
        self.biasGradient = 0
        self.batchSize = 0
            
class KAI:
    def __init__(self, inN, neuron, layer, outN, weights=[], learningRate=0.01):
        self.neuron = neuron
        self.layer = layer
        self.inN = inN
        self.outN = outN
        self.learningRate = learningRate
        self.previousWB = []
        self.model = []
        self.initiateModel(weights)
    def initiateModel(self, weights):
        self.model = []
        for l in range(len(self.layer)):
            self.model.append([])
            if l < len(self.layer) - 1:
                for n in range(self.neuron):
                    if weights == [] and l == 0:
                        self.model[l].append(Neuron(self.inN, self.layer[l]))
                    elif weights == []:
                        self.model[l].append(Neuron(self.neuron, self.layer[l]))
                    else:
                        self.model[l].append(Neuron(weights[l][n], self.layer[l]))
            else:
                for o in range(self.outN):
                    if weights == []:
                        self.model[l].append(Neuron(self.neuron, self.layer[l]))
                    else:
                        self.model[l].append(Neuron(weights[l][o], self.layer[l]))
    def calculate(self, inputs):
        self.inputs = inputs
        for l in range(len(self.layer)):
            self.outputs = []
            if l < len(self.layer) - 1:
                for n in range(self.neuron):
                    self.model[l][n].calculate(self.inputs)
                    self.outputs.append(self.model[l][n].output)
            else:
                for o in range(self.outN):
                    self.model[l][o].calculate(self.inputs)
                    self.outputs.append(self.model[l][o].output)
            self.inputs = self.outputs
    def distributeError(self, errorGradients):
        for l in reversed(range(len(self.layer))):
            if l >= len(self.layer) - 1:
                for o in range(self.outN):
                    self.model[l][o].delta = errorGradients[o] * self.model[l][o].activationDerivative()

                    for i in range(len(self.model[l][o].weight)):
                        self.model[l][o].weightGradients[i] += self.model[l][o].delta * self.model[l][o].inputs[i]
                    
                    self.model[l][o].biasGradient += self.model[l][o].delta
                    self.model[l][o].batchSize += 1
            else:
                for n in range(self.neuron):
                    total = 0
                    for nextNeuron in self.model[l+1]:
                        total+=nextNeuron.delta * nextNeuron.weight[n]

                    self.model[l][n].delta = total * self.model[l][n].activationDerivative()

                    for i in range(len(self.model[l][n].weight)):
                        self.model[l][n].weightGradients[i] += self.model[l][n].delta * self.model[l][n].inputs[i]
                    
                    self.model[l][n].biasGradient += self.model[l][n].delta
                    self.model[l][n].batchSize += 1
    def returnWB(self):
        return [
            [neuron.returnWB() for neuron in layer]
            for layer in self.model
        ]
    def updateWB(self):
        self.previousWB=self.returnWB()
        for layer in self.model:
            for neuron in layer:
                neuron.updateWB(self.learningRate)
    def rollback(self):
        self.initiateModel(self.previousWB)
        