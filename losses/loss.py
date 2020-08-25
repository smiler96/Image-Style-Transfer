import torch.nn as nn
from models.common.core import calc_mean_std

def content_loss(input, target, criterion=nn.MSELoss()):
    assert (input.size() == target.size())
    # assert (target.requires_grad is False)
    return criterion(input, target)

def style_loss(input, target, criterion=nn.MSELoss()):
    assert (input.size() == target.size())
    # assert (target.requires_grad is False)
    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)
    return criterion(input_mean, target_mean) + \
           criterion(input_std, target_std)
