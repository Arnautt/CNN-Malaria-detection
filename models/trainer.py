import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime

from utils.util import get_accuracy, get_f1_score, get_confusion_matrix, CheckPoint
from visualization.logger import TBLogger


class Trainer:
    """
    Class to train a Pytorch model for a vision task

    Parameters
    ----------
    model: nn.Module
        Model to be trained
    loader_train, loader_valid: torch Dataloader
        Respectively train and valid dataloader
    params_optim: dict
        Dictionary with a set of hyper-parameters for the optimizer
    device: torch.device
        Device to use (CPU or GPU)
    threshold: float in [0,1]
        Threshold to convert probability into label
    save_checkpoint_path: string
        Path where the checkpoints are saved
    verbose_freq: int
        Frequence where we show validation metrics during training
    save_log_path: string
        Path where the tensorboard logs are saved
    params_lr_scheduler: {dict, None}
        If given, dictionary with a set of hyper-parameters for the learning rate scheduler
    """
    def __init__(self, model, loader_train, loader_valid,
                 params_optim, device, threshold, save_checkpoint_path,
                 verbose_freq, save_log_path, params_lr_scheduler=None):
        self.model = model.to(device)
        self.params_optim = params_optim
        run_id = datetime.now().strftime(r'%d%m%Y_%H%M%S')
        self.save_checkpoint_path = save_checkpoint_path + "/checkpoint_" + run_id + ".pth"
        self.device = device
        self.verbose_freq = verbose_freq
        self.threshold = threshold
        self.loader_train = loader_train
        self.loader_valid = loader_valid
        self.use_lr_scheduler = (params_lr_scheduler is not None)
        self.optim = self._init_optimizer(self.model, params_optim)
        if self.use_lr_scheduler:
            self.scheduler = self._init_lr_scheduler(self.optim, params_lr_scheduler)
        self.loss_fn = self._init_loss()
        self.tb_logger = TBLogger(save_log_path)
        self.checkpoint = CheckPoint(self.save_checkpoint_path, self.model, self.optim)
        self.checkpoint.save()

    @staticmethod
    def _init_optimizer(model, params_optim):
        """Returns an optimizer given a dictionary of parameters and a model"""
        params_to_update = [p for p in model.parameters() if p.requires_grad]
        optim = torch.optim.SGD(params_to_update, **params_optim)
        return optim

    @staticmethod
    def _init_lr_scheduler(optim, params_lr_scheduler):
        """Returns a LR scheduler for a given optimizer and set of parameters"""
        # scheduler = torch.optim.lr_scheduler.CyclicLR(optim, **params_lr_scheduler)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, **params_lr_scheduler)
        return scheduler

    @staticmethod
    def _init_loss():
        """Initialize loss for our model"""
        loss_fn = nn.BCELoss(reduction='mean')
        return loss_fn

    def _train_one_epoch(self):
        """One epoch training phase"""
        train_loss = 0.
        self.model.train()
        for x, y in self.loader_train:
            x, y = x.to(self.device), y.to(self.device)
            self.optim.zero_grad()
            out = self.model(x)
            out = out.reshape(-1,)
            loss = self.loss_fn(out, y.float())
            train_loss += loss.item()
            loss.backward()
            self.optim.step()
            if self.use_lr_scheduler:
                self.scheduler.step()
        return train_loss

    def _validate_one_epoch(self):
        """Evaluate our model (accuracy, F1-score and loss) on validation data loader"""
        self.model.eval()
        conf_mat = get_confusion_matrix(model=self.model,
                                        threshold=self.threshold,
                                        data_loader=self.loader_valid,
                                        device=self.device)
        f1 = get_f1_score(conf_mat)
        acc = get_accuracy(conf_mat)
        valid_loss = 0.
        with torch.no_grad():
            for x, y in self.loader_valid:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                out = out.reshape(-1, )
                loss = self.loss_fn(out, y.float())
                valid_loss += loss.item()
        return acc, f1, valid_loss

    def train(self, n_iter):
        """Train model for n_iter iterations"""
        print("Start training, using device: {}".format(self.device))
        valid_best_f1, valid_best_acc = 0, 0
        for epoch in tqdm(range(n_iter)):
            train_loss = self._train_one_epoch()
            valid_acc, valid_f1, valid_loss = self._validate_one_epoch()
            if valid_acc > valid_best_acc:
                valid_best_acc = valid_acc
                self.checkpoint.save()
            if valid_f1 > valid_best_f1:
                valid_best_f1 = valid_f1
            self.tb_logger.log(train_loss, valid_loss, valid_acc, valid_f1, valid_best_acc, valid_best_f1, epoch)

            if epoch % self.verbose_freq == 0:
                kwargs = {"epoch": epoch, "val_acc": round(valid_acc, 4),
                          "val_f1": round(valid_f1, 4), "val_best_f1": round(valid_best_f1, 4)}
                to_print = "Epoch : {epoch}, validation accuracy = {val_acc}, f1-score = {val_f1}" \
                           " (max = {val_best_f1}),".format(**kwargs)
                print(to_print)

        self.checkpoint.load()
