from KAI import KAI
import json
from KAI_Visualizer import Visualizer
import math

#constants
inN = 1
neuronsPerLayer = 5
layers = [1, 1, 0] # 0: basic 1: Relu
outN = 1
errorThreshold = 0.01
errors = []
averageError = 1
previousError = 999999999999999999999999999
gen = 0
min_value = -10
max_value = 10
def func(x):
    return (
        0.2*x**2 
    )
def average(lis):
    return sum(lis) / len(lis)

model=KAI(inN,neuronsPerLayer,layers,outN)
visualizer = Visualizer(func, model, min_value, max_value)
while (averageError > errorThreshold) and (gen < 50000):
    errors=[]
    ques = [x / 10 for x in range(min_value * 10, max_value * 10)]
    ans = [func(q) for q in ques]
    visualizer.update()
    for q,a in zip(ques,ans):
        model.calculate([q])
        errors.append(0.5*(model.outputs[0]-a)**2)
        model.distributeError([model.outputs[0]-a])
    averageError = average(errors)
    if averageError > (1.05 + previousError):
        print("Generation "+ str(gen))
        print("Average Error Rate "+str(previousError))
        print("Failed Generation")
        model.rollback()
    else:
        previousError = averageError
        model.updateWB()
        print("Generation "+ str(gen))
        print("Average Error Rate "+str(averageError))
        print("Successful Generation")
    
    gen += 1
with open("KAIWB.txt", "w") as f:
    json.dump(model.returnWB(),f)
visualizer.close()