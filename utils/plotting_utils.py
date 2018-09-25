import torch



class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count





def count_parameters(model):
    """
    Count the trainable parameters of a model
    (Something to be plotted)
    """
    count = 0
    for p in model.parameters():
        if p.requires_grad:
            count += np.prod(p.size())
    return count


def parameter_magnitude(model):
    """
    Computes the (summed, absolute) magnitude of all parameters in a model
    (Something to be plotted)
    """
    mag = 0
    for p in model.parameters():
        if p.requires_grad:
            mag += torch.sum(torch.abs(p.data.cpu()))
    return mag

def gradient_magnitude(model):
    """
    Computes the (summed, absolute) magnitude of the current gradient
    (Something to be plotted)
    """
    mag = 0
    for p in model.parameters():
        if p.requires_grad:
            mag += torch.sum(torch.abs(p.grad.data.cpu()))
    return mag


def update_magnitude(model, lr, grad_magnitude=None):
    """
    Computes the magnitude of the current update
    (Something to be plotted)
    """
    if grad_magnitude is None:
        grad_magnitude = gradient_magnitude(model)
    return lr * grad_magnitude


def update_ratio(model, lr, params_magnitude=None, grad_magnitude=None):
    """
    Computes the ratio of the update magnitude relative to the parameter magnitudes
    We want to keep track of this and make sure that it doesn't get too large (otherwise we have unstable training)
    (Something to be plotted)
    """
    if params_magnitude is None:
        params_magnitude = parameter_magnitude(model)
    if grad_magnitude is None:
        grad_magnitude = gradient_magnitude(model)
    return lr * grad_magnitude / params_magnitude