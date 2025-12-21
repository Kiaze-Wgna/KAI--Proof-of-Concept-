import random

# constants
wMin=-1
wMax=1
bMin=0
bMax=1
class Neuron:
    def __init__(self,wb):
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
        self.output=self.bias
        for i,w in zip(inputs,self.weight):
            self.output+=i*w
        self.output=max(0,self.output)
    def returnWB(self):
        return [self.weight,self.bias]
    def randomizeWB(self,errorRate):
        self.weight=[
            random.uniform(
                max(wMin,w-(errorRate * (wMax - wMin))),
                min(wMax,w+(errorRate * (wMax - wMin)))
                )
            for w in self.weight
        ]
        self.bias=random.uniform(
            max(bMin,self.bias-(errorRate * (bMax - bMin))),
            min(bMax,self.bias+(errorRate * (bMax - bMin)))
            )

class KAI:
    def __init__(self,inN,neuron,layer,outN,weights=[]):
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
    def returnWB(self):
        return [
        [neuron.returnWB() for neuron in layer]
        for layer in self.model
        ]
    def randomizeWB(self,errorRate):
        for layer in self.model:
            for neuron in layer:
                neuron.randomizeWB(errorRate)

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