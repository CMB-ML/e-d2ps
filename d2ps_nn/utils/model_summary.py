# d2ps_nn/utils/model_summary.py
import torch

def trace_module_shapes(model, example_in_shape, dtype, device="cpu",
                        batch=1, leaves_only=True):
    """One dummy forward, capturing (name, class, in_shape, out_shape) per module.

    Shapes come from real tensors, so this stays correct across architecture
    changes instead of re-deriving conv arithmetic.
    """
    records, handles = [], []

    def make_hook(name):
        def hook(mod, inputs, output):
            in_shape = tuple(inputs[0].shape) if inputs and isinstance(inputs[0], torch.Tensor) else None
            out_shape = tuple(output.shape) if isinstance(output, torch.Tensor) else None
            records.append((name, type(mod).__name__, in_shape, out_shape))
        return hook

    for name, module in model.named_modules():
        if name == "":
            continue
        if leaves_only and any(module.children()):
            continue
        handles.append(module.register_forward_hook(make_hook(name)))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(torch.zeros(batch, *example_in_shape, dtype=dtype, device=device))
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    return records
