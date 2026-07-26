from KAI_Loss_and_Gradients_function import mse_loss, mse_gradients
from KAI_Basic_Constants import KAI_setup, min_value, max_value
from KAI import TrainingData

LINEAR = 0
LEAKY_RELU = 1
SIGMOID = 2
TANH = 3

inN = 2

layers = [
    [LEAKY_RELU, 5],
    [LEAKY_RELU, 5],
    [LINEAR, 1]
]

def func(x):
    return (
        0.2*x**2 
    )

trainingDataSet = []
for x in range(min_value * 10, max_value * 10):
    new_x = x / 10
    trainingDataSet.append(TrainingData(
        x_index = new_x, 
        inputs = [new_x * (-0.5), func(new_x)], 
        outputs = [(new_x * (-0.5)) + func(new_x)],
    ))

trainingPreset = KAI_setup(
    inN=inN,
    layers=layers,
    trainingDataSet=trainingDataSet,
    loss_function=mse_loss,
    gradients_function=mse_gradients
)