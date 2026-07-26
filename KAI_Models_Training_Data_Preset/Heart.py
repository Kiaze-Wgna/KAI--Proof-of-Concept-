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
    [TANH, 16],
    [TANH, 16],
    [LINEAR, 2]
]

trainingDataSet = []

for i in range(-100, 101):
    x = i / 100

    top = 3* (
        0.3 * math.sqrt(max(0, 1 - x*x))
        + 0.3 * abs(x)**0.35
        - 0.3
    )

    bottom = -1.45 * math.sqrt(max(0, 1 - abs(x)))

    trainingDataSet.append(
        TrainingData(
            x_index=x,
            inputs=[x],
            outputs=[top, bottom]
        )
    )

trainingPreset = KAI_setup(
    inN=inN,
    layers=layers,
    trainingDataSet=trainingDataSet,
    loss_function=mse_loss,
    gradients_function=mse_gradients,
    learningRate=0.01,
    batchSize=20,
    errorThreshold=0.01
)