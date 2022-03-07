import numpy as np
import torch


def tf_matrix(x, y, z, a, b, g):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor([a])
    if not isinstance(b, torch.Tensor):
        b = torch.tensor([b])
    if not isinstance(g, torch.Tensor):
        g = torch.tensor([g])
    ca = torch.cos(a)
    sa = torch.sin(a)
    cb = torch.cos(b)
    sb = torch.sin(b)
    cg = torch.cos(g)
    sg = torch.sin(g)
    tf = torch.zeros(4, 4, requires_grad=False)
    tf[0, 0], tf[0, 1], tf[0, 2], tf[0, 3] = cb*cg, sa*sb*cg - ca*sg, ca*sb*cg + sa*sg, x
    tf[1, 0], tf[1, 1], tf[1, 2], tf[1, 3] = cb*sg, sa*sb*sg + ca*cg, ca*sb*sg - sa*cg, y
    tf[2, 0], tf[2, 1], tf[2, 2], tf[2, 3] = -sb, sa*cb, ca*cb, z
    tf[3, 3] = 1
    return tf


def tf_inv(tf):
    return torch.inverse(tf)  # TF matrix inverse results in memory leaks, therefore torch.inverse

def zrot_matrix(theta):
    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor([theta])
    tf = torch.zeros(4, 4, requires_grad=False)
    tf[0, 0], tf[0, 1] = torch.cos(theta), -torch.sin(theta)
    tf[1, 0], tf[1, 1] = torch.sin(theta), torch.cos(theta)
    tf[2, 2], tf[3, 3] = 1, 1
    return tf
