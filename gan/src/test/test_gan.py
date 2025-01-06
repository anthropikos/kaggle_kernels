# Anthony Lee 2024-12-30

import unittest
import torch
from gan.gan import CycleGAN


class Test_CyCleGAN(unittest.TestCase):
    def test_parameters_iterator_notmpty(self):
        """Check that CycleGAN parameters iterator isn't empty.

        Had to mess with lazy modules for the model's submodule so that the
        parameters iterator does not return empty. This is crucial for the
        optimizers to know which parameters to keep track for autograd.
        """
        model = CycleGAN()
        param_iterator = model.parameters()
        param_list = list(param_iterator)

        self.assertNotEqual(len(param_list), 0)
        return


if __name__ == "__main__":
    unittest.main()
