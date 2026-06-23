from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from time_series_env.model_builder import ModelBuilder


def setup_device(use_gpu: bool) -> tuple:
    """Set up the compute device.

    PyTorch is optional for the core environment when a custom predictor is
    supplied. If it is not installed, the environment falls back to CPU mode.
    """
    try:
        import torch
    except ImportError:
        return "cpu", [], False

    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        device_ids = ",".join(str(i) for i in range(torch.cuda.device_count()))
        multi_gpu = torch.cuda.device_count() > 1
    else:
        device = torch.device("cpu")
        device_ids = []
        multi_gpu = False

    return device, device_ids, multi_gpu


def initialize_model(
    model_builder_class: type,
    device,
    device_ids: str,
    use_multi_gpu: bool,
) -> "ModelBuilder":
    """Initialize a model builder on the requested device."""
    model_builder = model_builder_class(
        device=device,
        device_ids=device_ids,
        use_multi_gpu=use_multi_gpu,
    )
    model_builder.to(device)

    return model_builder
