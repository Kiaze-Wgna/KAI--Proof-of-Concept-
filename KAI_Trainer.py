from KAI import KAI
import json
import random

#constants
inN=1
neuronsPerLayer=10
layers=2
outN=1
modelsPerGeneration=100
errorThreshold=0.001
errorRate=1
gen=0
def func(x):
    return (
        2.5*x
        + 1.0
    )
def average(lis):
    return sum(lis)/len(lis)

while errorRate>errorThreshold:
    ques=range(1,11)
    ans=[func(q) for q in ques]
    models=[]
    for i in range(modelsPerGeneration):
        if gen==0:
            model=KAI(inN,neuronsPerLayer,layers,outN)
        else:
            with open("KAIWB.txt", "r") as f:
                Wb=json.load(f)
            model=KAI(inN,neuronsPerLayer,layers,outN,Wb)
            model.randomizeWB(errorRate)
        errors=[]
        for q,a in zip(ques,ans):
            model.calculate([q])
            errors.append(abs(a-model.outputs[0])/abs(a))
        models.append([model,average(errors)])
    bestModel,errorRate= min(models, key=lambda x: x[1])
    if gen==0 or errorRate<previousErrorRate:
        previousBestModel=bestModel
        previousErrorRate=errorRate
        print("Successful Generation")
    else:
        bestModel=previousBestModel
        errorRate=previousErrorRate
        print("Failed Generation")
    with open("KAIWB.txt", "w") as f:
        json.dump(bestModel.returnWB(),f)
    print("Generation "+str(gen)+":")
    print("Error Rate: "+str(errorRate))
    gen+=1