import unittest
import torch
from gan.discriminator import Discriminator
from gan.downsampler import Downsampler
from gan.upsampler import Upsampler


class Test_Discriminator(unittest.TestCase):

    def test_intput_dim_greater(self):
        """Test batched input when dimension is greater than 256."""
        input = torch.randint(255, size=(10, 3, 257, 257)).float()
        layer = Discriminator(downsampler_factory=Downsampler)
        with self.assertRaises(ValueError) as cm:
            output = layer(input)
        return

    def test_input_dim_lesser(self):
        """Test batched input when dimension is lesser than 256."""
        input = torch.randint(255, size=(10, 3, 255, 255)).float()
        layer = Discriminator(downsampler_factory=Downsampler)
        with self.assertRaises(ValueError) as cm:
            output = layer(input)
        return

    def test_unbatched_input(self):
        """Input should be batched"""
        input_size = (3, 256, 256)
        input = torch.randint(255, size=input_size).float()
        layer = Discriminator(downsampler_factory=Downsampler)
        with self.assertRaises(ValueError) as cm:
            output = layer(input)

        return

    def test_check_output_shape(self):
        input = torch.randint(low=0, high=10, size=(5, 3, 256, 256), dtype=torch.float32)
        layer = Discriminator(downsampler_factory=Downsampler)
        output = layer(input)
        self.assertEqual(output.size(), torch.Size((5, 1, 30, 30)))


if __name__ == "__main__":
    unittest.main()
