import os
import torch

from utils.dataset import get_dataloaders, sample_data
from utils.util import load_config, get_confusion_matrix, get_f1_score, get_accuracy
from models.pretrained import build_pretrained_model
from models.model import MyCNN
from models.trainer import Trainer
from visualization.explainer import Explainer


def main(config):
    # Load data loaders and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader_train, loader_test, loader_valid = get_dataloaders(**config["dataloader"])

    if config["use_pretrained_model"]:
        model = build_pretrained_model(config["model_name"])
    else:
        model = MyCNN(im_size=config["dataloader"]["im_size"],
                      config_block=config["model"]["block"],
                      n_out_channels=config["model"]["n_out_channels"],
                      n_hidden_neurons=config["model"]["n_hidden_neurons"])

    # Train model
    trainer = Trainer(model=model,
                      loader_train=loader_train,
                      loader_valid=loader_valid,
                      params_optim=config["params_optim"],
                      params_lr_scheduler=None,  # config["params_lr_scheduler"],
                      device=device,
                      **config["trainer"])

    trainer.train(n_iter=config["train"]["n_iter"])

    # Test model on new observations (to see model's performance in a real life scenario)
    conf_mat_test = get_confusion_matrix(model=trainer.model,
                                         threshold=config["trainer"]["threshold"],
                                         data_loader=loader_test,
                                         device=device)
    f1_test = round(get_f1_score(conf_mat_test), 3)
    acc_test = round(get_accuracy(conf_mat_test), 3)
    print("On the test set, F1 score = {} and accuracy = {}".format(f1_test, acc_test))

    # Explain some predictions with SHAP values
    e = Explainer(model, loader_train, device, savepath="./saved/figures")
    images_test, label_test = sample_data(loader_test, 5)
    e.fit()
    e.explain(images_test, label_test)


if __name__ == "__main__":
    CONFIG_PATH = os.getcwd()
    CONFIG_NAME = "config.yaml"
    config = load_config(CONFIG_PATH, CONFIG_NAME)
    main(config)
