"""Fixed statistical regional coordinates; no vehicle-conservation claim."""
import torch
from torch import nn
import torch.nn.functional as F


class RegionGeometry(nn.Module):
    def __init__(self, assignment):
        super().__init__()
        ids = torch.as_tensor(assignment, dtype=torch.long)
        if ids.ndim != 1 or ids.numel() == 0 or int(ids.min()) != 0:
            raise ValueError('Expected nonempty zero-based region assignment')
        regions = int(ids.max())+1
        counts = torch.bincount(ids, minlength=regions)
        if bool((counts == 0).any()):
            raise ValueError('Regions must be contiguous and nonempty')
        s = F.one_hot(ids, num_classes=regions).float()
        self.register_buffer('assignment', ids)
        self.register_buffer('S', s)
        self.register_buffer('P', s.T/counts[:, None])
        self.register_buffer('counts', counts)
        self.nodes, self.regions = len(ids), regions

    def pool(self, x):
        if x.shape[-2] != self.nodes:
            raise ValueError('Node axis does not match fixed partition')
        return torch.matmul(self.S.T.to(dtype=x.dtype), x)/self.counts.to(dtype=x.dtype)[:, None]

    def lift(self, x):
        if x.shape[-2] != self.regions:
            raise ValueError('Region axis does not match fixed partition')
        return torch.matmul(self.S.to(dtype=x.dtype), x)

    def center(self, x):
        return x-self.lift(self.pool(x))

    def additive_coordinates(self, reference, prediction, scale):
        delta = (prediction-reference)/scale
        return self.pool(delta), self.center(delta)


def pool_contrast(geometry, local, weights):
    return geometry.pool(local*weights)


def lift_contrast(geometry, regional, weights):
    return geometry.center(geometry.lift(regional)*weights)
