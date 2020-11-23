import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


class TBLogger:
    """
    Class for saving train/test loss & metrics with tensorboard

    Parameters
    ----------
    path: string
        Path where the results are saved
    """
    def __init__(self, path):
        dir_name = path + "/log_" + datetime.now().strftime(r'%d%m%Y_%H%M%S')
        os.makedirs(dir_name)
        self.writer = SummaryWriter(dir_name)
        print("Tensorboard logs path: {}".format(dir_name))

    def log(self, train_loss, valid_loss, acc_test, f1_test, best_acc, best_f1, iteration):
        """Log losses and metrics to SummaryWriter"""
        self.writer.add_scalars('Loss', {"train": train_loss, "valid": valid_loss}, iteration)
        self.writer.add_scalars('Validation/F1', {"F1": f1_test, "MAX_F1": best_f1}, iteration)
        self.writer.add_scalars('Validation/Accuracy', {"acc": acc_test, "max_acc": best_acc}, iteration)
