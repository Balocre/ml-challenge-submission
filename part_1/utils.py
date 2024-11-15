import os
import tempfile

import torch


def calculate_model_size(model):
    """A function to calculate the size of a torch model in memory and on disk

    :param model: The input model

    :return: The size of the model on disk and in memory in MB
    """

    def get_size(torch_object):
        if torch_object.data.is_floating_point():
            return torch_object.numel() * torch.finfo(torch_object.data.dtype).bits
        else:
            return torch_object.numel() * torch.iinfo(torch_object.data.dtype).bits

    parameters_size = 0
    for param in model.parameters():
        parameters_size += get_size(param)

    buffers_size = 0
    for buffer in model.buffers():
        buffers_size += get_size(buffer)

    model_size_memory = parameters_size + buffers_size

    with tempfile.NamedTemporaryFile("wb") as f:
        torch.save(model, f)
        model_size_disk = os.stat(f.name).st_size / (1024**2)

    return model_size_disk, model_size_memory / 8e6


def build_id2cls_map(words_path):
    """A function that builds a dict mapping the ids from imagenet to human-readable classes

    :param words_path: The path to the words.txt file

    :return: A dictionary with imagenet class ids as keys and human-readable classes as values
    """

    with open(words_path) as f:
        id2cls_map = dict()
        for line in f.readlines():
            id, cls = line.split("\t")
            id2cls_map[id] = cls

    return id2cls_map


def build_imagenet1000_classes(im1000_words_path):
    """A function that build a list of classes fro ImageNet-1000

    Each classes index in the list corresponds to the class number

    :param im100_words_path: The path to the words.txt file

    :return: A list of class IDs
    """

    with open(im1000_words_path, "r") as f:
        im1000_classes = list()
        for line in f.readlines():
            id, _ = line.split("\t")
            im1000_classes.append(id)

    return im1000_classes


def quantize_model(model, dtype=torch.qint8):
    """A wrapper function to perform dynamic quantization of a torch model

    It is set to only quantize Linear layers.

    :param model: The model to quantize
    :param dtype: The quantization type

    :return: The quantized model
    """

    model_q = torch.quantization.quantize_dynamic(
        model, qconfig_spec={torch.nn.Linear}, dtype=dtype
    )

    return model_q
