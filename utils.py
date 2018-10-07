import numpy as np

def check_path_name(file_path):
    if file_path[-1] != "/":
        return file_path+"/"
    else:
        return file_path


def num_params(model) :
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3f million' % parameters)