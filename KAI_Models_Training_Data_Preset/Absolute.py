from KAI_Loss_and_Gradients_function import mse_loss, mse_gradients
from KAI_Basic_Constants import KAI_setup
from KAI import TrainingData


LINEAR = 0
LEAKY_RELU = 1
SIGMOID = 2
TANH = 3


inN = 1

layers = [
    [LEAKY_RELU, 8],
    [LEAKY_RELU, 8],
    [LINEAR, 1],
]


def func(x):
    return abs(x)


trainingDataSet=[]

for i in range(-100,100):
    x=i/10

    trainingDataSet.append(
        TrainingData(
            x_index=x,
            inputs=[x],
            outputs=[func(x)]
        )
    )


trainingPreset = KAI_setup(
    inN=inN,
    layers=layers,
    trainingDataSet=trainingDataSet,
    loss_function=mse_loss,
    gradients_function=mse_gradients,
    size_per_generation=1,
    errorThreshold=0.001
)