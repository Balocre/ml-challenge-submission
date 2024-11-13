import torch

def calculate_model_size(model):
    """ A function to calculate the size of a torch model
    
    :param model: The input model
    
    :return: The size of the model in bit and MB

    """

    size_model = 0
    for param in model.parameters():
        if param.data.is_floating_point():
            size_model += param.numel() * torch.finfo(param.data.dtype).bits
        else:
            size_model += param.numel() * torch.iinfo(param.data.dtype).bits
        
    return size_model, size_model / 8e6