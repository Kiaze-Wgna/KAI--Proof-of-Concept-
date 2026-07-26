from dataclasses import dataclass
from typing import Callable
from KAI_evaluation_functions import nmse

@dataclass
class KAI_setup:
    inN: int
    layers: list
    trainingDataSet: list
    loss_function: Callable
    gradients_function: Callable
    evaluation_function: Callable = nmse
    errorThreshold: float = 0.01
    maxGeneration: int = 1000
    size_per_generation: int = 5
    batchSize: int = 20
    learningRate: float = 0.01
    momentum: float = 0.0
min_value = -10
max_value = 10