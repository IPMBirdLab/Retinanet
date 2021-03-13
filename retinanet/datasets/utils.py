import numpy as np
from torch.utils.data import Dataset


def train_val_split(dataset, p=0.8, use_p_of_data=1):
    """Splits dataset into train and validation by calculating
    indices corresponding to each subset.

    Parameters
    ----------
    dataset : torch.Dataset
        Dataset object which its len() give the size of dataset
    p : float, optional
        percentage (a number between 0-1) of training samples, by default 0.8
    use_p_of_data : int, optional
        percentage (a number between 0-1) of dataset to use, by default 1

    Returns
    -------
    tuple[list, list]
        lists of indices for training samples and validation samples
    """
    if p > 1 or p < 0:
        return None
    length = int(np.ceil(use_p_of_data * len(dataset)))
    split = int(np.ceil(p * length))
    index_list = list(range(length))

    train_idx, val_idx = index_list[:split], index_list[split:]
    return train_idx, val_idx


class TransformDatasetWrapper(Dataset):
    """Wrap dataset to have seperate transfrmation for train and test

    implementation from:
    https://discuss.pytorch.org/t/why-do-we-need-subsets-at-all/49391/8
    https://discuss.pytorch.org/t/using-imagefolder-random-split-with-multiple-transforms/79899
    https://discuss.pytorch.org/t/apply-different-transform-data-augmentation-to-train-and-validation/63580

    Parameters
    ----------
    Dataset : torch.utils.data.Dataset
    """

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        # Load actual image here
        x = self.dataset[index]
        if self.transform:
            x = self.transform(*x)

        return x

    def __len__(self):
        return len(self.dataset)


def one_hot_embedding(label, classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (int) class labels.
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """

    vector = np.zeros((classes), dtype=np.float32)
    if len(label) > 0:
        vector[label] = 1.0
    return vector
