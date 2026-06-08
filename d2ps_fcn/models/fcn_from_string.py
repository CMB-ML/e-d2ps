import re
import torch
import torch.nn as nn


class PixelPairProductLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outer = torch.bmm(x.unsqueeze(2), x.unsqueeze(1))
        return outer.flatten(start_dim=1)


class Square(nn.Module):
    def forward(self, x):
        return x * x


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
    "Q": Square,
}


class Residual(nn.Module):
    """
    ResNet-style block:

        out = post_act(F(x) + skip(x))

    If in_features == out_features, skip is identity.
    Otherwise, skip is a bias-free linear projection.

    gated=True gives ReZero behavior:

        out = post_act(skip(x) + alpha * F(x))

    with alpha initialized to 0.
    """

    def __init__(self, block, in_features, out_features, post_act=None, gated=False):
        super().__init__()
        self.block = block

        if in_features == out_features:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Linear(in_features, out_features, bias=False)

        self.post_act = post_act if post_act is not None else nn.Identity()
        self.alpha = nn.Parameter(torch.zeros(())) if gated else None

    def forward(self, x):
        fx = self.block(x)

        if self.alpha is not None:
            fx = self.alpha * fx

        return self.post_act(fx + self.skip(x))


class FCNFromString(nn.Module):
    """
    Compact architecture grammar.

    Examples
    --------
    Plain network:

        N_768R_768R_O

    Residual block:

        N_768R_a768R_768Rz_O

    Multiple residual blocks:

        N_768R_a768R_768Rz_a768R_768Rz_O

    Pixel-pair input:

        P_768R_a768R_768Rz_O

    Gated/ReZero residual block:

        N_768R_a768R_768R+z_O

    Token meanings
    --------------
    N      : normal input
    P      : pixel-pair product input
    768R   : Linear(..., 768) + ReLU
    768G   : Linear(..., 768) + GELU
    O      : output layer, no activation
    OR     : output layer + ReLU
    a768R  : start residual block with first layer 768R
    768Rz  : end residual block with final layer 768R
    768R+z : end gated/ReZero residual block
    """

    def __init__(self, arch_str, n_in, n_out):
        super().__init__()

        tokens = arch_str.split("_")
        if len(tokens) < 2:
            raise ValueError("arch string needs an input token and an output token.")

        head, *rest = tokens
        if head not in ("N", "P"):
            raise ValueError(f"first token must be 'N' or 'P', got '{head}'.")

        self.pixel_pair = head == "P"

        in_features = n_in ** 2 if self.pixel_pair else n_in
        self.example_in_shape = (in_features,)
        self.example_out_shape = (n_out,)

        groups = self._group(rest)

        layers = []
        if self.pixel_pair:
            layers.append(PixelPairProductLayer())

        prev = in_features

        for gi, (kind, payload) in enumerate(groups):
            is_last_group = gi == len(groups) - 1

            if kind == "plain":
                dim, act = self._parse_layer_token(payload, is_last_group, n_out)
                layers.append(nn.Linear(prev, dim))

                if not isinstance(act, nn.Identity):
                    layers.append(act)

                prev = dim

            elif kind == "res":
                block_toks, gated = payload
                res, prev = self._build_residual_block(block_toks, prev, gated)
                layers.append(res)

            else:
                raise RuntimeError(f"unknown group kind {kind}")

        self.net = nn.Sequential(*layers)
        self.model_dict = None

    @staticmethod
    def _group(tokens):
        """
        Converts tokens into:

            ("plain", token)

        or

            ("res", ([tokens_inside_block], gated))
        """
        groups = []
        buf = None
        gated = False

        for raw_tok in tokens:
            if raw_tok == "":
                raise ValueError("empty token found in architecture string.")

            starts = raw_tok.startswith("a")
            ends = raw_tok.endswith("z")

            tok = raw_tok

            if starts:
                if buf is not None:
                    raise ValueError("nested residual blocks are not supported.")
                tok = tok[1:]
                buf = []

            if ends:
                tok = tok[:-1]

            if tok.endswith("+"):
                gated = True
                tok = tok[:-1]

            if tok == "":
                raise ValueError(f"invalid structural token '{raw_tok}'.")

            if buf is not None:
                buf.append(tok)

                if ends:
                    if len(buf) < 2:
                        raise ValueError(
                            "residual block should contain at least two layer tokens."
                        )
                    groups.append(("res", (buf, gated)))
                    buf = None
                    gated = False

            else:
                if ends:
                    raise ValueError(f"stray block end marker in token '{raw_tok}'.")
                groups.append(("plain", tok))

        if buf is not None:
            raise ValueError("unclosed residual block: missing trailing 'z'.")

        return groups

    def _build_residual_block(self, toks, in_features, gated):
        """
        ResNet-style convention:

            Linear -> Act -> Linear -> add skip -> final Act

        So the activation from the LAST token is moved after the residual add.
        """
        layers = []
        prev = in_features
        post_act = nn.Identity()

        for i, tok in enumerate(toks):
            dim, act = self._parse_layer_token(tok, is_last=False, n_out=None)
            layers.append(nn.Linear(prev, dim))

            if i == len(toks) - 1:
                post_act = act
            else:
                if not isinstance(act, nn.Identity):
                    layers.append(act)

            prev = dim

        block = nn.Sequential(*layers)
        return Residual(block, in_features, prev, post_act=post_act, gated=gated), prev

    @staticmethod
    def _parse_layer_token(tok, is_last, n_out):
        if is_last:
            if tok == "O":
                return n_out, nn.Identity()

            if len(tok) == 2 and tok[0] == "O" and tok[1] in ACTIVATIONS:
                return n_out, ACTIVATIONS[tok[1]]()

            raise ValueError(f"last token must be O or O<act>, got '{tok}'.")

        match = re.fullmatch(r"(\d+)([A-Z])", tok)
        if not match:
            raise ValueError(
                f"layer token '{tok}' must look like 768R, 512G, 64T, etc."
            )

        dim = int(match.group(1))
        act_key = match.group(2)

        if act_key not in ACTIVATIONS:
            raise ValueError(f"unknown activation '{act_key}' in token '{tok}'.")

        return dim, ACTIVATIONS[act_key]()

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    torch.manual_seed(0)

    n_in, n_out, B = 12, 5, 8
    x = torch.randn(B, n_in)
    y = torch.randn(B, n_out)

    cases = [
        "N_50R_50R_O",
        "N_768R_a768R_768Rz_O",
        "N_768R_a768R_768Rz_a768R_768Rz_O",
        "N_a64R_32Rz_a32R_16Rz_O",
        "N_768R_a768R_768R+z_O",
        "P_20R_a20R_20Rz_O",
    ]

    for c in cases:
        m = FCNFromString(c, n_in, n_out)
        out = m(x)

        assert out.shape == (B, n_out), (c, out.shape)

        skips = [
            type(mod.skip).__name__
            for mod in m.modules()
            if isinstance(mod, Residual)
        ]

        gated = [
            mod.alpha is not None
            for mod in m.modules()
            if isinstance(mod, Residual)
        ]

        print(f"OK  {c:45s} out={tuple(out.shape)} skips={skips} gated={gated}")