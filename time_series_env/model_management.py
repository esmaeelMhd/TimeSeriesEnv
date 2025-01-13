import torch
from time_series_env.model_builder import ModelBuilder

def setup_device(use_gpu: bool) -> tuple:
    """Setup the device based on GPU usage."""    
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        device_ids = ','.join(str(i) for i in range(torch.cuda.device_count()))
        multi_gpu = len(device_ids) > 1
    else:
        device = torch.device('cpu')
        device_ids = []
        multi_gpu = False
    return device, device_ids, multi_gpu
    
def initialize_model(model_builder_class: type, device: torch.device, device_ids: str, use_multi_gpu: bool) -> ModelBuilder:
    """Initialize the model based on the provided model builder class."""
    model_builder = model_builder_class(device=device, device_ids=device_ids, use_multi_gpu=use_multi_gpu)
    model_builder.to(device)
    
    return model_builder
