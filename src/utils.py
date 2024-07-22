import numpy as np
import torch
import torch.nn as nn


class bcolors:
    """ Author: Omar
    This class helps to print colored text in the terminal.
    To uce this class, call cprint(text, color)
    """
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RED = '\033[91m'
    FAIL = '\033[91m'
    OKGREEN = '\033[92m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    ORANGE = '\033[93m'
    BLUE = '\033[94m'
    HEADER = '\033[95m'
    CYAN = '\033[96m'


def cprint(text, color: str = "", end='\n'):
    """ Author: Omar
    Colorful print function. To see the colors, go to the class bcolors.

    Usage:
    cprint('You may fluff her tai', bcolors.OKGREEN)
    cprint('Warning: no sadge allowed', bcolors.WARNING)
    cprint('Failed to be sadge', bcolors.FAIL)
    """
    print(color + text + bcolors.ENDC, end=end)


def bar(p, color=bcolors.RED, length=20, do_write_p=False):
    """plot a progress bar of given color"""
    p = min(1, max(0, p))
    chars = '▏▎▍▌▋▊▉██'
    n = int(p * length)
    residue = p * length - n
    out = f"{'█'*n}"
    if residue:
        i = int(residue * 8)
        out += f'{chars[i]}'
    out = out.ljust(length, ' ')
    out = f"{color}{out}{bcolors.ENDC}"
    out = f"[{out}]"
    if do_write_p:
        out += f' {p:7.1%}'
    return out


def initialize_weights(model, weight_range=0.1):
    """
    Initialize weights and biases of the model
    within the given range (in-place operation)
    """
    assert weight_range > 0, "Weight range must be positive"
    for layer in model.children():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.uniform_(layer.weight, -weight_range, weight_range)
            torch.nn.init.uniform_(layer.bias, -weight_range, weight_range)


def binary_array_to_indices(a):
    """Convert a binary array to a list of indices"""
    return np.where(a)[0]


class WeightedMSELoss(nn.Module):
    def __init__(self, weight=1.):
        self.weight = weight
        super(WeightedMSELoss, self).__init__()

    def forward(self, input, target):
        # Calculate the weights based on the target probabilities
        weights = 1 + (2 * target - 1) ** 2 * self.weight
        # Compute the weighted MSE loss
        loss = weights * (input - target) ** 2
        return loss.mean()
