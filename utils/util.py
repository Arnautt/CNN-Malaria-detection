import numpy as np
import torch
import yaml
import os


class CheckPoint:
    """
    Class for saving state of a model and its optimizer during training phase.

    Parameters
    ----------
    checkpoint_path: string
        Path where the checkpoints are saved
    model: torch.nn.Module
        Model to be saved
    optim: torch.nn.optim
        Optimizer to be used during training
    """
    def __init__(self, checkpoint_path, model, optim):
        self.checkpoint_path = checkpoint_path
        self.model = model
        self.optim = optim

    def load(self):
        """Load last saved optimizer and model states"""
        last_checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(last_checkpoint['model_state_dict'])
        self.optim.load_state_dict(last_checkpoint['optimizer_state_dict'])

    def remove(self):
        """Remove saved checkpoint"""
        os.remove(self.checkpoint_path)

    def save(self):
        """Save current state of optimizer and model"""
        to_be_saved = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
        }
        torch.save(to_be_saved, self.checkpoint_path)


def load_config(path, config_name):
    """Load YAML configuration file given its path and file name"""
    with open(os.path.join(path, config_name)) as file:
        config = yaml.safe_load(file)
    return config


def get_confusion_matrix(model, threshold, data_loader, device):
    """
    Calculate a confusion matrix for binary classification of a Pytorch model.

    Parameters
    ----------
    model: torch.nn.Module
        Pytorch model to be tested
    threshold: float in [0,1]
        Threshold to use to convert probability into label
    data_loader: torch Dataloader
        DataLoader to be tested. Need to have shuffle attribute to False.
    device: torch.device
        Device to use (CPU or GPU)

    Returns
    -------
    conf_mat: numpy.ndarray
        2x2 numpy confusion matrix
    """
    conf_mat = np.zeros((2, 2), dtype=int)
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            y_pred = (out >= threshold).reshape(-1,)
            tp = ((y == 1) & (y_pred == 1)).sum().item()
            tn = ((y == 0) & (y_pred == 0)).sum().item()
            fp = ((y == 0) & (y_pred == 1)).sum().item()
            fn = ((y == 1) & (y_pred == 0)).sum().item()
            conf_mat[0, 0] += tp
            conf_mat[1, 1] += tn
            conf_mat[0, 1] += fn
            conf_mat[1, 0] += fp
    return conf_mat


def get_f1_score(conf_mat):
    """
    Compute F1-score for binary classification, given a confusion matrix.

    Parameters
    ---------
    conf_mat: numpy.ndarray
        Numpy confusion matrix of shape 2x2

    Returns
    -------
    f1: float in [0,1]
        F1-score
    """
    tp = conf_mat[0, 0].item()
    fn, fp = conf_mat[0, 1].item(), conf_mat[1, 0].item()
    try:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1 = 0
    return f1


def get_accuracy(conf_mat):
    """
    Compute accuracy of a model given its confusion matrix

    Parameters
    ----------
    conf_mat: numpy.ndarray
        Numpy confusion matrix of shape 2x2

    Returns
    -------
    acc: float in [0,1]
        Accuracy of the model
    """
    true_classified = conf_mat[0, 0] + conf_mat[1, 1]
    n_samples = conf_mat.sum()
    acc = true_classified / n_samples
    return acc
