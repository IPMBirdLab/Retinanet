try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from torch import nn


def logits_to_preds(logits):
    return (logits > 0.5).float()


def outputs_to_logits(outputs):
    return nn.Sigmoid()(outputs)
