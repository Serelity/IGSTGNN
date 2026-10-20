"""Reuse the certified train/validation adapter; no GW17 test loader exists."""
from experiments.position_state_gw16.data import load_data, device_batch
from .common import require


def validation_loader(data, split, batch_size, logger):
    require(split == 'val', 'GW17 development has no test evaluation entry')
    loader = data.loaders['val_loader']
    return loader, 3909
