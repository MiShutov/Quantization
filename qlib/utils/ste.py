import torch

def ste_round_pass(x):
    return x.round().detach() - x.detach() + x

