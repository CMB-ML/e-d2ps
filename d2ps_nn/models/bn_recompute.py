import logging

import torch
import torch.nn as nn
from tqdm import tqdm


logger = logging.getLogger(__name__)


@torch.no_grad()
def recompute_bn_stats(model, dataloader, device, dtype, max_batches=None):
    """
    Recompute BatchNorm running statistics from scratch over `dataloader`,
    using the model's CURRENT weights.

    Buffers are reset and then accumulated with a cumulative average
    (momentum=None), so the result is the exact mean/var over the batches
    seen -- not an EMA. This removes both the initialization bias (buffers
    starting at 0/1) and the recency weighting of an EMA.

    The model is placed in eval mode so dropout is inactive; only the
    BatchNorm modules are flipped to train mode, since buffers are only
    updated in train mode. Original modes and momenta are restored on exit.

    Returns the number of batches used.
    """
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return 0

    was_training = model.training
    saved = [(m.momentum, m.training) for m in bns]

    model.eval()                      # dropout off for the whole model
    for m in bns:
        m.reset_running_stats()       # zeroes buffers AND num_batches_tracked
        m.momentum = None             # cumulative average over just this pass
        m.train()                     # buffers only update in train mode

    n = 0
    try:
        total = len(dataloader) if max_batches is None else min(max_batches, len(dataloader))
        for features, _ in tqdm(dataloader, total=total, desc="BN recompute", leave=False):
            if max_batches is not None and n >= max_batches:
                break
            model(features.to(device=device, dtype=dtype))
            n += 1
    finally:
        for m, (mom, m_training) in zip(bns, saved):
            m.momentum = mom
            m.train(m_training)
        model.train(was_training)

    return n
