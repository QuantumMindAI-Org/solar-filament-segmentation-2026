"""Soft clDice (Shit et al. style) with the pre-registered pooling skeleton. Pure torch, no state.

All tensors are (B, 1, H, W) floats in [0, 1]. Min-pool is implemented as -max_pool(-x); max_pool pads with
-inf so image borders are neutral (they do not erode and do not create foreground).
"""
import torch
from torch.nn import functional as F

ITERATIONS = 8
SMOOTH = 1.0


def soft_erode(x):
    """Minimum of a 3x1 and a 1x3 min-pool."""
    vertical = -F.max_pool2d(-x, (3, 1), stride=1, padding=(1, 0))
    horizontal = -F.max_pool2d(-x, (1, 3), stride=1, padding=(0, 1))
    return torch.minimum(vertical, horizontal)


def soft_dilate(x):
    """3x3 max-pool."""
    return F.max_pool2d(x, 3, stride=1, padding=1)


def soft_open(x):
    return soft_dilate(soft_erode(x))


def soft_skeleton(x, iterations=ITERATIONS):
    skel = F.relu(x - soft_open(x))
    for _ in range(iterations):
        x = soft_erode(x)
        delta = F.relu(x - soft_open(x))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice(probability, target, iterations=ITERATIONS, smooth=SMOOTH):
    """Per-sample soft clDice loss (1 - clDice), shape (B,); the caller averages.

    tprec = (sum(skel(p)*y)+1)/(sum(skel(p))+1); tsens = (sum(skel(y)*p)+1)/(sum(skel(y))+1).
    The target skeleton carries no gradient.
    """
    axes = (1, 2, 3)
    skel_p = soft_skeleton(probability, iterations)
    with torch.no_grad():
        skel_y = soft_skeleton(target, iterations)
    tprec = ((skel_p * target).sum(axes) + smooth) / (skel_p.sum(axes) + smooth)
    tsens = ((skel_y * probability).sum(axes) + smooth) / (skel_y.sum(axes) + smooth)
    return 1 - 2 * tprec * tsens / (tprec + tsens)
