from KAI_Loss_and_Gradients_function import mse_loss, mse_gradients
from KAI_Basic_Constants import KAI_setup
from KAI import TrainingData
import random


LINEAR = 0
LEAKY_RELU = 1
SIGMOID = 2
TANH = 3


inN = 2

layers = [
    [LEAKY_RELU, 8],
    [LINEAR, 1],
]


trainingDataSet=[]

for i in range(200):
    a=random.uniform(-10,10)
    b=random.uniform(-10,10)

    trainingDataSet.append(
        TrainingData(
            x_index=i,
            inputs=[a,b],
            outputs=[a+b]
        )
    )


trainingPreset = KAI_setup(
    inN=inN,
    layers=layers,
    trainingDataSet=trainingDataSet,
    loss_function=mse_loss,
    gradients_function=mse_gradients,
)