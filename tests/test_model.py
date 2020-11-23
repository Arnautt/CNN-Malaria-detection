import unittest
import torch

from models.model import ConvBlock, MyCNN


class TestModelFromScratch(unittest.TestCase):
    """Test model trained from scratch"""
    def test_shape_after_forward_pass(self):
        conv_block = ConvBlock(in_channels=3,
                               in_height=224,
                               in_width=224,
                               out_channels=5,
                               kernel_size=2,
                               stride=2,
                               probs_dropout=0.1)
        batch_images = torch.randn(10, 3, 224, 224)
        out = conv_block(batch_images)
        expected_size = [10, 5, 56, 56]
        self.assertEqual(expected_size, list(out.shape))

    def test_theoretical_shape(self):
        conv_block = ConvBlock(in_channels=3,
                               in_height=224,
                               in_width=224,
                               out_channels=5,
                               kernel_size=2,
                               stride=2,
                               probs_dropout=0.1)
        expected_h_out, expected_w_out = 56, 56
        h_out = conv_block.h_out
        w_out = conv_block.w_out
        self.assertEqual(expected_h_out, h_out)
        self.assertEqual(expected_w_out, w_out)

    def test_model_output_shape(self):
        config_block = {'kernel_size': 2, 'stride': 2, 'probs_dropout': 0.15}
        n_out_channels = [3, 10, 15]
        n_hidden_neurons = [512, 256]
        model = MyCNN(im_size=224,
                      config_block=config_block,
                      n_out_channels=n_out_channels,
                      n_hidden_neurons=n_hidden_neurons)
        batch_images = torch.randn(10, 3, 224, 224)
        out = model(batch_images)
        self.assertEqual(list(out.shape), [10, 1])


if __name__ == '__main__':
    unittest.main()
