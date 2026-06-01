import torch
import torch.nn as nn


class PixelPairProductLayer(nn.Module):
    """
    Computes the flattened outer product of a pixel map with itself.
    For input x of shape (batch, N), produces output of shape (batch, N*N).

    This encodes all pairwise products x_i * x_j, giving the network
    direct access to the two-point correlation structure.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, N)
        outer = torch.bmm(x.unsqueeze(2), x.unsqueeze(1))  # (batch, N, N)
        return outer.flatten(start_dim=1)                   # (batch, N*N)


class Square(nn.Module):
    """Element-wise square. Turns a linear projection into a squared projection
    so that linear->Square->linear realises a sum of squared projections, i.e.
    any quadratic form (variance, power spectrum). Unbounded -- pair with a
    relative/whitened loss to avoid scale domination."""
    def forward(self, x):
        return x * x


# Maps the single-character code in an "a<code>" suffix to a constructor.
ACTIVATIONS = {
    "R": nn.ReLU,
    "G": nn.GELU,
    "S": nn.Sigmoid,
    "T": nn.Tanh,
    "I": nn.Identity,
    "L": nn.LeakyReLU,
    "W": nn.SiLU,
    "E": nn.ELU,
    "P": nn.Softplus,
    "Q": Square
}


class FCNFromString(nn.Module):
    """
    Build an FCN from a compact arch string.

    Grammar (tokens separated by underscores, one per Linear layer):
      - First token is the input spec:
          'N'  -> plain input of width n_in
          'P'  -> pixel-pair mode; input is expanded to n_in**2 up front
                  via the flattened self outer product.
      - Last token is the output: 'O' (Identity) or 'O<act>' (dimension n_out).
      - Every middle token is '<dim><act>', e.g. '50aR'.
      - Activations are mandatory on hidden layers; use 'aI' for Identity.
        On the output, the activation is optional and defaults to Identity.

    Example: 'P_50aR_50aR_O'
      pixel-pair(n_in**2) -> Linear(->50) -> ReLU
                          -> Linear(->50) -> ReLU
                          -> Linear(->n_out)
    """

    def __init__(self, arch_str: str, n_in: int, n_out: int):
        super().__init__()

        tokens = arch_str.split("_")
        if len(tokens) < 2:
            raise ValueError(
                "arch string needs at least an input token and an output token."
            )

        head, *rest = tokens
        if head not in ("N", "P"):
            raise ValueError(f"first token must be 'N' or 'P', got '{head}'.")
        self.pixel_pair = head == "P"

        in_features = n_in ** 2 if self.pixel_pair else n_in
        self.example_in_shape = {in_features}
        self.example_out_shape = {n_out}

        layers: list[nn.Module] = []
        if self.pixel_pair:
            layers.append(PixelPairProductLayer())

        prev = in_features
        for i, tok in enumerate(rest):
            is_last = i == len(rest) - 1
            dim, act = self._parse_token(tok, is_last, n_out)
            layers.append(nn.Linear(prev, dim))
            if not isinstance(act, nn.Identity):
                layers.append(act)
            prev = dim

        self.net = nn.Sequential(*layers)
        self.model_dict = None  # TODO

    @staticmethod
    def _parse_token(tok: str, is_last: bool, n_out: int):
        """Split a '<dim><act>' (or output 'O' / 'O<act>') token into (dim, act)."""
        idx = tok.find("a")
        dim_part = tok if idx == -1 else tok[:idx]
        act_part = "" if idx == -1 else tok[idx:]

        # Resolve the activation. Bare (no 'a...') is only allowed on the
        # output, where it means Identity; hidden layers must specify one.
        if act_part == "":
            if not is_last:
                raise ValueError(
                    f"token '{tok}' is missing a mandatory activation (e.g. 'aR')."
                )
            act = nn.Identity()
        else:
            if len(act_part) != 2 or act_part[1] not in ACTIVATIONS:
                raise ValueError(
                    f"token '{tok}' has an invalid activation '{act_part}'. "
                    f"valid codes: {''.join(sorted(ACTIVATIONS))}."
                )
            act = ACTIVATIONS[act_part[1]]()

        # Resolve the dimension.
        if is_last:
            if dim_part != "O":
                raise ValueError(f"last token must be 'O' or 'O<act>', got '{tok}'.")
            dim = n_out
        else:
            if dim_part == "O":
                raise ValueError("'O' output token can only appear last.")
            if not dim_part.isdigit():
                raise ValueError(f"token '{tok}' has a non-integer dim '{dim_part}'.")
            dim = int(dim_part)

        return dim, act

    def forward(self, x):
        return self.net(x)
