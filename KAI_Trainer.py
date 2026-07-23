from KAI import KAI
import json
from KAI_Visualizer import Visualizer
import math
import random
from dataclasses import dataclass

#constants
inN = 2
layers = [[1, 5], [1, 5], [0, 1]] # 0: basic 1: Relu
errorThreshold = 0.1
errors = []
averageError = 1
previousError = 999999999999999999999999999
gen = 0
batchSize = 20
min_value = -10
max_value = 10

@dataclass
class TrainingData:
    x_index: int
    inputs: list
    outputs: list

def func(x):
    return (
        0.2*x**2 
    )
def average(lis):
    return sum(lis) / len(lis)

def _get_loss(type, prediction, actual):
    if len(prediction) != len(actual):
        raise ValueError(f"Expected {len(actual)} outputs, but received {len(prediction)} outputs.")
    loss = 0
    for p, a in zip(prediction, actual):
        #Types: 0:MeanSquaredError
        if type == 0:
            loss += 0.5 * (p - a) **2
    return loss
    
def _get_gradients(type, prediction, actual):
    if len(prediction) != len(actual):
        raise ValueError(f"Expected {len(actual)} outputs, but received {len(prediction)} outputs.")
    gradients = []
    for p, a in zip(prediction, actual):
        #Types: 0:MeanSquaredError
        if type == 0:
            gradients.append(p-a)
    return gradients

model=KAI(inN,layers)
trainingDataSet = []
for x in range(min_value * 10, max_value * 10):
    new_x = x / 10
    trainingDataSet.append(TrainingData(
        x_index = new_x, 
        inputs = [new_x * (-0.1), func(new_x)], 
        outputs = [(new_x * (-0.1)) + func(new_x)],
    ))

visualizer = Visualizer(trainingDataSet, model)
while (averageError > errorThreshold) and (gen < 5000):
    errors=[]
    random.shuffle(trainingDataSet)
    visualizer.update()
    currentSample = 0
    for trainingData in trainingDataSet:
        model.calculate(trainingData.inputs)
        errors.append(_get_loss(0, model.outputs, trainingData.outputs))
        model.distributeError(_get_gradients(0, model.outputs, trainingData.outputs))
        currentSample += 1
        if currentSample == batchSize:
            currentSample = 0
            model.updateWB()
    if currentSample != 0:
        model.updateWB()
    averageError = average(errors)
    print("Generation "+ str(gen))
    print("Average Error Rate "+str(average(errors)))
    
    gen += 1
with open("KAIWB.txt", "w") as f:
    json.dump(model.returnWB(),f)
visualizer.close()