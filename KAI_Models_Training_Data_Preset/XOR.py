from KAI_Loss_and_Gradients_function import bce_loss, bce_gradients
from KAI_evaluation_functions import bce_accuracy
from KAI_Basic_Constants import KAI_setup
from KAI import TrainingData

LINEAR = 0
LEAKY_RELU = 1
SIGMOID = 2
TANH = 3

inN = 2

layers = [
    [LEAKY_RELU, 4],
    [SIGMOID, 1],
]

trainingDataSet = [
    TrainingData(
        x_index=0,
        inputs=[0, 0],
        outputs=[0]
    ),
    TrainingData(
        x_index=1,
        inputs=[0, 1],
        outputs=[1]
    ),
    TrainingData(
        x_index=2,
        inputs=[1, 0],
        outputs=[1]
    ),
    TrainingData(
        x_index=3,
        inputs=[1, 1],
        outputs=[0]
    ),
]

trainingPreset = KAI_setup(
    inN=inN,
    layers=layers,
    trainingDataSet=trainingDataSet,
    loss_function=bce_loss,
    gradients_function=bce_gradients,
    evaluation_function=bce_accuracy,
    errorThreshold=0.05,
    learningRate=0.5,
    batchSize=1
)