import torch
import torch.nn as nn
import torch.nn.functional as F


class _AttenuatedBNMixin:
    """
    Interpolates between running-stat normalization and batch-stat normalization.

    alpha = 1.0 : ordinary BatchNorm (full batch-stat noise)
    alpha = 0.0 : normalize by running stats only (no batch-draw noise)
    alpha = a   : z_run + a * (z_batch - z_run)

    Running stats are still UPDATED every training step regardless of alpha,
    so z_run stays calibrated as the non-BN weights drift. Freeze them
    separately by setting momentum=0 on the module if you want them fixed.

    At eval time (self.training == False) this is exactly nn.BatchNorm:
    it normalizes by running stats via super().forward().
    """

    def __init__(self, num_features, *args, alpha=0.1, **kwargs):
        super().__init__(num_features, *args, **kwargs)
        if not self.track_running_stats:
            raise ValueError("AttenuatedBatchNorm requires track_running_stats=True")
        self.alpha = float(alpha)

    def forward(self, x):
        self._check_input_dim(x)

        # Eval: identical to ordinary BatchNorm (running-stat normalization + affine).
        if not self.training:
            return super().forward(x)

        # Deterministic branch: normalize by running stats, BEFORE they are updated.
        # weight/bias=None so affine is applied once, later, to the blended result.
        z_run = F.batch_norm(
            x, self.running_mean, self.running_var,
            weight=None, bias=None,
            training=False, momentum=0.0, eps=self.eps,
        )

        # Replicate PyTorch's running-stat update bookkeeping exactly.
        if self.num_batches_tracked is not None:
            self.num_batches_tracked.add_(1)
            if self.momentum is None:               # cumulative moving average
                factor = 1.0 / float(self.num_batches_tracked)
            else:
                factor = self.momentum
        else:
            factor = self.momentum

        # Stochastic branch: batch-stat normalization; this call ALSO updates
        # running_mean / running_var in place (training=True).
        z_batch = F.batch_norm(
            x, self.running_mean, self.running_var,
            weight=None, bias=None,
            training=True, momentum=factor, eps=self.eps,
        )

        z = z_run + self.alpha * (z_batch - z_run)

        if self.affine:
            shape = (1, -1) + (1,) * (x.ndim - 2)
            z = z * self.weight.view(*shape)
            if self.bias is not None:
                z = z + self.bias.view(*shape)
        return z


class AttenuatedBatchNorm1d(_AttenuatedBNMixin, nn.BatchNorm1d):
    pass


class AttenuatedBatchNorm2d(_AttenuatedBNMixin, nn.BatchNorm2d):
    pass


def set_bn_alpha(model, alpha):
    """Set alpha on every AttenuatedBatchNorm module in a model."""
    alpha = float(alpha)
    for m in model.modules():
        if isinstance(m, _AttenuatedBNMixin):
            m.alpha = alpha
