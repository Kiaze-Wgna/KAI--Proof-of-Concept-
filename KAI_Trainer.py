from KAI import KAI
import json
from KAI_Visualizer import Visualizer
import math
import random

#constants
inN = 1
layers = [[1, 5], [1, 5], [0, 1]] # 0: basic 1: Relu
errorThreshold = 0.1
errors = []
averageError = 1
previousError = 999999999999999999999999999
gen = 0
batchSize = 20
min_value = -10
max_value = 10
def func(x):
    return (
        0.2*x**2 
    )
def average(lis):
    return sum(lis) / len(lis)

def _get_loss(type, prediction, actual):
    #Types: 0:MeanSquaredError
    if type == 0:
        return 0.5 * (prediction - actual) **2
    
def _get_gradients(type, prediction, actual):
    #Types: 0:MeanSquaredError
    if type == 0:
        return [prediction - actual]

model=KAI(inN,layers)
visualizer = Visualizer(func, model, min_value, max_value)
while (averageError > errorThreshold) and (gen < 5000):
    errors=[]
    trainingData = []
    for x in range(min_value * 10, max_value * 10):
        ques = x / 10
        trainingData.append([ques, func(ques)])
    random.shuffle(trainingData)
    visualizer.update()
    currentSample = 0
    for qa in trainingData:
        model.calculate([qa[0]])
        errors.append(_get_loss(0, model.outputs[0], qa[1]))
        model.distributeError(_get_gradients(0, model.outputs[0], qa[1]))
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