#########################################################################
#                    Normalized Mean Squared Error                      #
#########################################################################
def nmse(model, trainingDataSet):
    predictions = []
    actuals = []

    for trainingData in trainingDataSet:
        model.calculate(trainingData.inputs)
        predictions.extend(model.outputs)
        actuals.extend(trainingData.outputs)
    mean_actual = sum(actuals) / len(actuals)

    variance = sum(
        (a - mean_actual) ** 2
        for a in actuals
    ) / len(actuals)

    mse = sum(
        (p - a) ** 2
        for p, a in zip(predictions, actuals)
    ) / len(predictions)

    if variance == 0:
        return mse

    return mse / variance
#########################################################################
#                              Accuracy                                 #
#########################################################################
def accuracy(model, trainingDataSet):
    correct = 0
    total = 0

    for trainingData in trainingDataSet:
        model.calculate(trainingData.inputs)
        for prediction, actual in zip(
            model.outputs,
            trainingData.outputs
        ):
            predicted_class = 1 if prediction >= 0.5 else 0
            if predicted_class == actual:
                correct += 1
            total += 1

    return 1 - (correct / total)

#########################################################################
#                         Binary Cross Entropy                          #
#########################################################################
from KAI_Loss_and_Gradients_function import bce_loss
def bce_accuracy(model, trainingDataSet):
    errors = []
    for trainingData in trainingDataSet:
        model.calculate(trainingData.inputs)
        errors.append(bce_loss(model.outputs, trainingData.outputs))
    return sum(errors) / len(errors)