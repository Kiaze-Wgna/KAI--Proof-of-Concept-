import math
EPSILON = 1e-12

#########################################################################
#                          Mean Squared Error                           #
#########################################################################
def mse_loss(prediction, actual):
    if len(prediction) != len(actual):
        raise ValueError(f"Expected {len(actual)} outputs, but received {len(prediction)} outputs.")
    loss = 0
    for p, a in zip(prediction, actual):
        loss += 0.5 * (p - a) **2
    return loss
    
def mse_gradients(prediction, actual):
    if len(prediction) != len(actual):
        raise ValueError(f"Expected {len(actual)} outputs, but received {len(prediction)} outputs.")
    gradients = []
    for p, a in zip(prediction, actual):
        gradients.append(p-a)
    return gradients
#########################################################################
#                         Binary Cross Entropy                          #
#########################################################################
def bce_loss(prediction, actual):
    if len(prediction) != len(actual):
        raise ValueError(f"Expected {len(actual)} outputs, but received {len(prediction)} outputs.")
    loss = 0
    for p, a in zip(prediction, actual):
        p = max(EPSILON, min(1 - EPSILON, p))
        loss -= (a * math.log(p)) + ((1 - a)*math.log(1 - p))
    return loss
    
def bce_gradients(prediction, actual):
    if len(prediction) != len(actual):
        raise ValueError(f"Expected {len(actual)} outputs, but received {len(prediction)} outputs.")
    gradients = []
    for p, a in zip(prediction, actual):
        #p = max(EPSILON, min(1 - EPSILON, p))
        #gradients.append(((1 - a) / (1 - p)) - (a / p))
        gradients.append(p-a)
    return gradients