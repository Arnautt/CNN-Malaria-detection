import os
import shap
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from utils.dataset import un_normalize


class Explainer:
    """
    SHAP Explainer class for vision Pytorch models

    Parameters
    ----------
    model: nn.Module
        Model to be explained
    loader_train: DataLoader
        Pytorch DataLoader to fit explainer
    device: torch.device
        Device to use (CPU or GPU)
    n_train_sample: int
        Number of training samples to use to fit explainer
    local_smoothing: float
        Local smoothing hyper-parameter
    savepath: {None, string}
        If given, save plot at this path
    """
    def __init__(self, model, loader_train, device, n_train_sample=100, local_smoothing=1, savepath=None):
        self.model = model.to(device)
        self.loader_train = loader_train
        self.device = device
        self.n_train_sample = n_train_sample
        self.local_smoothing = local_smoothing
        background_data, _ = self.sample_data(loader_train, n_train_sample)
        self.background_data = background_data.to(device)
        self.savepath = savepath
        self.id = datetime.now().strftime(r'%d%m%Y_%H%M%S')
        self.e = None

    @staticmethod
    def sample_data(data_loader, n_to_sample):
        """Sample n_to_sample data from a Pytorch DataLoader"""
        x, y = next(iter(data_loader))
        data = x[:n_to_sample]
        labels = y[:n_to_sample]
        return data, labels

    def fit(self):
        """Fit the SHAP explainer"""
        self.e = shap.GradientExplainer(self.model, self.background_data, self.local_smoothing)

    def explain(self, images, labels):
        """Explain normalized images of shape N_test * channels * height * width with corresponding labels"""
        # Calculate SHAP values and reshape for image_plot
        images = images.to(self.device)
        probs = self.model(images).reshape(-1,)
        shap_values = self.e.shap_values(images)
        shap_values = [np.stack(shap_values, axis=0)]
        shap_values = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
        images_numpy = un_normalize(images, self.device).cpu().numpy()
        images_numpy = np.swapaxes(np.swapaxes(images_numpy, 1, -1), 1, 2)

        # Add prediction for each images
        _ = shap.image_plot(shap_values, images_numpy, show=False)
        fig = plt.gcf()
        all_axes = fig.get_axes()
        for i in range(len(all_axes) - 1):
            if i % 2 == 0:
                prob = probs.data.cpu().numpy()[i // 2]
                label = labels.data.cpu().numpy()[i // 2]
                title = "True label: {} ; model's probability to be infected: {:.2f}".format(label, prob)
                all_axes[i].set_title(title)
        plt.subplots_adjust(bottom=0.25)

        # Plot or save explained images
        if self.savepath is not None:
            path = os.path.join(self.savepath, 'explained_images_{}.png'.format(self.id))
            plt.savefig(path, bbox_inches='tight')
        else:
            plt.show()
