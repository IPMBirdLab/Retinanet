try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import torch
from torch import nn

from collections import OrderedDict
from typing import Dict


def logits_to_preds(logits):
    return (logits > 0.5).float()


def outputs_to_logits(outputs):
    return nn.Sigmoid()(outputs)


#######################################################################
# Load Model State Dictionary
#######################################################################
# Dinamically loading model weights
def get_map_location():
    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = "cpu"

    return map_location


def merge_state_dicts(model_std, pretrained_std):
    merged_dict = {}
    for k, v in model_std.items():
        if k in pretrained_std and v.size() == pretrained_std[k].size():
            merged_dict[k] = pretrained_std[k]
        else:
            merged_dict[k] = v

    return merged_dict


def load_chpt(model, source):
    if isinstance(source, OrderedDict):
        state_dict = source
    elif isinstance(source, str):
        state_dict = torch.load(source)

    std = model.state_dict()
    state_dict = merge_state_dicts(std, state_dict)
    model.load_state_dict(state_dict)

    return model


#######################################################################
