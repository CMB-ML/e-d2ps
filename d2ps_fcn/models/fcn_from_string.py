import torch.nn as nn


class FCNFromString(nn.Module):
    def __init__(self, arch_str: str, npix: int, n_ells: int):
        super().__init__()
        dims = self._parse(arch_str, npix, n_ells)
        layers = []
        for i, (in_d, out_d) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(in_d, out_d))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def _parse(self, arch_str: str, nside: int, n_ells: int) -> list[int]:
        parts = arch_str.split("_")
        dims = []
        for i, p in enumerate(parts):
            if p == "N":
                dims.append(nside if i == 0 else n_ells)
            else:
                dims.append(int(p))
        return dims

    def forward(self, x):
        return self.net(x)
