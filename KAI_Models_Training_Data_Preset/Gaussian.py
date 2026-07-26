from KAI_Loss_and_Gradients_function import mse_loss, mse_gradients
from KAI_Basic_Constants import KAI_setup
from KAI import TrainingData
import math


LINEAR = 0
LEAKY_RELU = 1
SIGMOID = 2
TANH = 3


inN = 1


layers = [
    [LEAKY_RELU, 10],
    [LEAKY_RELU, 10],
    [LINEAR, 1],
]


def gaussian(x):
    return math.exp(-(x*x)/5)


trainingDataSet = []


index = 0

for i in range(-100,100):

    x = i/10

    trainingDataSet.append(
        TrainingData(
            x_index=x,
            inputs=[x],
            outputs=[gaussian(x)]
        )
    )

    index += 1


trainingPreset = KAI_setup(
    inN=inN,
    layers=layers,
    trainingDataSet=trainingDataSet,
    loss_function=mse_loss,
    gradients_function=mse_gradients,
)