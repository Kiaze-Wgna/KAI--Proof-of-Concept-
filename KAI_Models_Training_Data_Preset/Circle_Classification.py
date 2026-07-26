from KAI_Loss_and_Gradients_function import bce_loss, bce_gradients
from KAI_evaluation_functions import accuracy
from KAI_Basic_Constants import KAI_setup
from KAI import TrainingData
import random
import math


LINEAR = 0
LEAKY_RELU = 1
SIGMOID = 2
TANH = 3


inN = 2

layers = [
    [LEAKY_RELU, 8],
    [LEAKY_RELU, 8],
    [SIGMOID, 1],
]


trainingDataSet = []

index = 0

for _ in range(500):

    x = random.uniform(-10,10)
    y = random.uniform(-10,10)

    distance = math.sqrt(x*x + y*y)

    label = 1 if distance < 5 else 0


    trainingDataSet.append(
        TrainingData(
            x_index=index,
            inputs=[x,y],
            outputs=[label]
        )
    )

    index += 1


trainingPreset = KAI_setup(
    inN=inN,
    layers=layers,
    trainingDataSet=trainingDataSet,
    loss_function=bce_loss,
    gradients_function=bce_gradients,
    evaluation_function=accuracy,
    learningRate=0.1,
    batchSize=20
)