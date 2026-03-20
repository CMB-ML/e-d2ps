import torch


class SingleDWishartLoss(torch.nn.Module):
    def __init__(self, ells):
        super().__init__()

        ells = torch.tensor(ells, dtype=torch.float32)
        # ells = torch.arange(0, lmax + 1, dtype=torch.float32)
        # ells = ells[2:]
        ratio = (2 * ells + 1) / (2 * ells - 1)
        self.register_buffer("ratio", ratio)

    def forward(self, output, target):
        t1 = output / target
        t2 = torch.log(target)
        t3 = torch.log(output)

        per_ell = t1 + t2 - self.ratio * t3

        return torch.sum(per_ell)

def get_loss(loss_key, ells):
    if loss_key == "l1":
        loss_function = torch.nn.L1Loss(reduction='mean')
    elif loss_key == "l2":
        loss_function = torch.nn.MSELoss(reduction='mean')
    elif loss_key == "wishart":
        loss_function = SingleDWishartLoss(ells=ells)
    else:
        raise ValueError(f"Unknown loss function: {loss_key}")
    return loss_function
