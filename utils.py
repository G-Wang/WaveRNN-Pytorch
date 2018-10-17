import numpy as np

def num_params(model) :
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    print('Trainable Parameters: %.3f million' % parameters)