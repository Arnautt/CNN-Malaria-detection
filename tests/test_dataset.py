import torch
import unittest
import numpy as np
from torchvision import transforms

from utils.dataset import un_normalize


class TestDataset(unittest.TestCase):
    """Test dataset functions"""
    def test_normalize(self):
        batch_images = torch.randint(low=0, high=255, size=(3, 224, 224)) / 255.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        tfm_test = transforms.Normalize(mean=mean, std=std)
        batch_images_transformed = tfm_test(batch_images)
        batch_images_untransformed = un_normalize(batch_images_transformed, device='cpu')
        vec1 = np.round(batch_images.data.numpy(), 2)
        vec2 = np.round(batch_images_untransformed.data.numpy(), 2)
        tst = np.all(vec1 == vec2)
        self.assertTrue(tst)


if __name__ == '__main__':
    unittest.main()
