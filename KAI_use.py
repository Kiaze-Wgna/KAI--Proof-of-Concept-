from KAI import KAI
import json

inN=1
neuronsPerLayer=10
layers=2
outN=1
with open("KAIWB.txt", "r") as f:
    Wb=json.load(f)
    model=KAI(inN,neuronsPerLayer,layers,outN,Wb)
model.calculate([999])
print(model.outputs)