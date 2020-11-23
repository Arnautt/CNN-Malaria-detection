import unittest
import torch.nn as nn
import torch

from models.pretrained import build_pretrained_model


class TestWeightsUpdate(unittest.TestCase):
    """Test if weights update for different models"""
    def test_pretrained_alexnet(self):
        loss_fn = nn.BCELoss(reduction='mean')
        model = build_pretrained_model(model_name="alexnet")
        params_to_update = [p for p in model.parameters() if p.requires_grad]
        optim = torch.optim.SGD(params_to_update, lr=0.5)
        old_params = [p.clone() for p in model.parameters()]
        x = torch.randn(10, 3, 224, 224)
        y = torch.zeros(10)
        out = model(x)
        out = out.reshape(-1, )
        loss = loss_fn(out, y.float())
        loss.backward()
        optim.step()
        new_params = [p.clone() for p in model.parameters()]
        tst = any([(old_p != new_p).byte().any().item() for old_p, new_p in zip(old_params, new_params)])
        self.assertTrue(tst)

    def test_pretrained_squeezenet(self):
        loss_fn = nn.BCELoss(reduction='mean')
        model = build_pretrained_model(model_name="squeezenet")
        params_to_update = [p for p in model.parameters() if p.requires_grad]
        optim = torch.optim.SGD(params_to_update, lr=0.5)
        old_params = [p.clone() for p in model.parameters()]
        x = torch.randn(10, 3, 224, 224)
        y = torch.zeros(10)
        out = model(x)
        out = out.reshape(-1, )
        loss = loss_fn(out, y.float())
        loss.backward()
        optim.step()
        new_params = [p.clone() for p in model.parameters()]
        tst = any([(old_p != new_p).byte().any().item() for old_p, new_p in zip(old_params, new_params)])
        self.assertTrue(tst)


if __name__ == '__main__':
    unittest.main()
