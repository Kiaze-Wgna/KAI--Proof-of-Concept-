import random

# constants
wMin=-1
wMax=1
bMin=0
bMax=1

def average(lis):
    return sum(lis)/len(lis)

class Neuron:
    def __init__(self,wb):
        self.inputs=[]
        self.ogOutput=0
        self.output=0
        self.gradients=[]
        if type(wb) is list:
            self.weight=wb[0]
            self.bias=wb[1]
        else:
            self.weight=[
                random.uniform(wMin,wMax)
                for w in range(wb)
            ]
            self.bias=random.uniform(bMin,bMax)
    def calculate(self,inputs):
        self.inputs=inputs

        self.ogOutput=self.bias
        for i,w in zip(inputs,self.weight):
            self.ogOutput+=i*w
        self.output=max(0,self.ogOutput)
    def returnWB(self):
        return [self.weight,self.bias]
    def updateWB(self,learningRate):
        for w in range(len(self.weight)):
            self.weight[w]-= learningRate*average(self.gradients)*self.inputs[w]
        self.bias-= learningRate*average(self.gradients)
            
class KAI:
    def __init__(self,inN,neuron,layer,outN,weights=[],learningRate=0.01):
        self.neuron=neuron
        self.layer=layer
        self.outN=outN
        self.model=[]
        for l in range(self.layer):
            self.model.append([])
            for n in range(self.neuron):
                if weights==[] and l==0:
                    self.model[l].append(Neuron(inN))
                elif weights==[]:
                    self.model[l].append(Neuron(neuron))
                else:
                    self.model[l].append(Neuron(weights[l][n]))
        self.model.append([])
        for o in range(self.outN):
            if weights==[]:
                self.model[self.layer].append(Neuron(neuron))
            else:
                self.model[self.layer].append(Neuron(weights[self.layer][o]))
        self.learningRate=learningRate
    def calculate(self,inputs):
        self.inputs=inputs
        for l in range(self.layer):
            self.outputs=[]
            for n in range(self.neuron):
                self.model[l][n].calculate(self.inputs)
                self.outputs.append(self.model[l][n].output)
            self.inputs=self.outputs
        self.outputs=[]
        for o in range(self.outN):
            self.model[self.layer][o].calculate(self.inputs)
            self.outputs.append(self.model[self.layer][o].output)
        self.inputs=self.outputs
    def distributeError(self, errorGradients):
        for o in range(self.outN):
            self.model[self.layer][o].gradient=errorGradients[o]
            
            if self.model[self.layer][o].ogOutput<=0:
                self.model[self.layer][o].gradients.append(0)
            else:
                self.model[self.layer][o].gradients.append(errorGradients[o])
        for l in reversed(range(self.layer)):
            for n in range(self.neuron):
                if self.model[l][n].ogOutput<=0:
                    self.model[l][n].gradients.append(0)
                else:
                    total=0
                    for nextNeuron in self.model[l+1]:
                        total+=nextNeuron.gradients[-1] * nextNeuron.weight[n]
                    self.model[l][n].gradients.append(total)

    def returnWB(self):
        return [
        [neuron.returnWB() for neuron in layer]
        for layer in self.model
        ]
    def updateWB(self):
        for layer in self.model:
            for neuron in layer:
                neuron.updateWB(self.learningRate)
                neuron.gradients=[]

#weights=[[ [[1],5], [[2],5], [[3],5] ],
#         [ [[1,2,3],5], [[1,2,3],5], [[1,2,3],5] ],
#         [ [[1,2,3],5], [[1,2,3],5]]]
#kai=KAI(1,3,2,2)
#kai.calculate([5])
#print(kai.returnWB())
#print(kai.outputs)
#print()
#kai.randomizeWB(0.5)
#kai.calculate([5])
#print(kai.returnWB())
#print(kai.outputs)