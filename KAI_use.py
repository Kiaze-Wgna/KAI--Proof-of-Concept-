from KAI import KAI
import json

inN=1
layers = [[1, 5], [1, 5], [0, 1]]
with open("KAIWB.txt", "r") as f:
    Wb=json.load(f)
    model=KAI(inN,layers,Wb)
model.calculate([3])
print(model.outputs)