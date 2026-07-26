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
    [TANH, 14],
    [TANH, 14],
    [LINEAR, 2]
]

trainingDataSet=[]

for i in range(-100,101):
    x=i/100

    upper=math.sqrt(max(0,1-x*x))+0.25*math.sin(8*x)
    lower=-math.sqrt(max(0,1-x*x))-0.25*math.sin(8*x)

    trainingDataSet.append(
        TrainingData(
            x_index=x,
            inputs=[x],
            outputs=[upper,lower]
        )
    )

trainingPreset=KAI_setup(
    inN=inN,
    layers=layers,
    trainingDataSet=trainingDataSet,
    loss_function=mse_loss,
    gradients_function=mse_gradients,
    errorThreshold=0.001
)