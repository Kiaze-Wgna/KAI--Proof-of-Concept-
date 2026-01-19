from KAI import KAI
import json
import math

#constants
inN=1
neuronsPerLayer=10
layers=2
outN=1
errorThreshold=0.1
errors=[]
averageError=1
gen=0
def func(x):
    return (
        0.1*x**2+2*x+2
    )
def average(lis):
    return sum(lis)/len(lis)

model=KAI(inN,neuronsPerLayer,layers,outN)

while (averageError>errorThreshold)and (gen<50000):
    ques=range(1,4)
    ans=[func(q) for q in ques]
    for q,a in zip(ques,ans):
        model.calculate([q])
        errors.append(0.5*(model.outputs[0]-a)**2)
        model.distributeError([model.outputs[0]-a])
    averageError=average(errors)
    model.updateWB()
    print("Generation "+ str(gen))
    print("Average Error Rate "+str(averageError))
    gen+=1
with open("KAIWB.txt", "w") as f:
    json.dump(model.returnWB(),f)