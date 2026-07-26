from KAI_Loss_and_Gradients_function import mse_loss, mse_gradients
from KAI_Basic_Constants import KAI_setup
from KAI import TrainingData

LINEAR = 0
LEAKY_RELU = 1
SIGMOID = 2
TANH = 3


def func(x):
    return 0.05 * x**3


inN = 1

layers = [
    [LEAKY_RELU, 8],
    [LEAKY_RELU, 8],
    [LINEAR, 1],
]


trainingDataSet = []

for x in range(-100,100):
    value = x / 10
    trainingDataSet.append(
        TrainingData(
            x_index=value,
            inputs=[value],
            outputs=[func(value)]
        )
    )


trainingPreset = KAI_setup(
    inN=inN,
    layers=layers,
    trainingDataSet=trainingDataSet,
    loss_function=mse_loss,
    gradients_function=mse_gradients,
)