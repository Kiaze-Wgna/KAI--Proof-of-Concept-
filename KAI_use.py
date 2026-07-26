from KAI import KAI
import json

with open("KAIWB.txt", "r") as f:
    inN, layers, wb, learningRate, momentum=json.load(f)
    model=KAI(inN, layers, wb, learningRate, momentum)
model.calculate([3])
print(model.outputs)