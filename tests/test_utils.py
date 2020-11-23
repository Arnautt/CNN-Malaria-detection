import unittest
import numpy as np

from utils.util import get_f1_score


class TestUtils(unittest.TestCase):
    """Test utility functions"""
    def test_f1_score_zero_division(self):
        conf_mat = np.array([[0, 20], [0, 15]])
        f1 = get_f1_score(conf_mat)
        tst = (f1 == 0)
        self.assertTrue(tst)


if __name__ == '__main__':
    unittest.main()
