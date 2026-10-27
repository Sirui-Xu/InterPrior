"""Loading helpers for checkpoints and prepared motion tensors."""

import inspect
from pathlib import Path


def load_tensor_file(path):
    """Load a trusted PyTorch file with the installed InterMimic environment."""
    import torch

    options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = False
    return torch.load(str(Path(path)), **options)
