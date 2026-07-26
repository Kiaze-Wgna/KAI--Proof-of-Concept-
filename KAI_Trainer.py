from KAI import KAI
import json
from KAI_Visualizer import Visualizer

# Presets Import
from KAI_Models_Training_Data_Preset.Parabola import trainingPreset

errors = []

models=[]
for _ in range(trainingPreset.size_per_generation):
    models.append(KAI(
        inN=trainingPreset.inN,
        layer=trainingPreset.layers,
        learningRate=trainingPreset.learningRate,
        momentum=trainingPreset.momentum
    ))

gen = 0
visualizer = Visualizer(trainingPreset.trainingDataSet, models)
bestError = 99999999999999999999999999
while (bestError > trainingPreset.errorThreshold) and (gen < trainingPreset.maxGeneration):
    losses = []
    errors = []
    for model in models:
        losses.append(model.train(trainingPreset.trainingDataSet, trainingPreset.batchSize, trainingPreset.loss_function, trainingPreset.gradients_function))
        errors.append(trainingPreset.evaluation_function(model, trainingPreset.trainingDataSet))
    visualizer.update(errors)
    bestError = min(errors)
    print("Generation "+ str(gen))
    print("Best Loss Rate "+str(min(losses)))
    print("Best Error Rate "+str(bestError))
    gen += 1
best_model_index = errors.index(bestError)
bestModel = models[best_model_index]
with open("KAIWB.txt", "w") as f:
    json.dump(bestModel.storeModel(),f)
visualizer.close(best_model_index)