import unittest
import torch
from gan.generator import Generator
from gan.downsampler import Downsampler
from gan.upsampler import Upsampler


class Test_Generator(unittest.TestCase):
    def test_check_output_dimension(self):
        """The dimension of the output of the generator is expected to be the same as the input."""
        intput_dim = (10, 3, 256, 256)
        input = torch.randint(255, size=intput_dim, dtype=torch.float32)
        layer = Generator(downsampler_factory=Downsampler, upsampler_factory=Upsampler)
        output = layer(input)

        self.assertEqual(output.size(), torch.Size(intput_dim))
        return

    def test_input_greater_dim(self):
        """Test batched input when dimension is greater than 256."""
        input = torch.randint(255, size=(10, 3, 257, 257)).float()
        layer = Generator(downsampler_factory=Downsampler, upsampler_factory=Upsampler)
        with self.assertRaises(ValueError) as cm:
            output = layer(input)
        return

    def test_input_smaller_dim(self):
        """Test batched input when dimension is lesser than 256."""
        input = torch.randint(255, size=(10, 3, 255, 255)).float()
        layer = Generator(downsampler_factory=Downsampler, upsampler_factory=Upsampler)
        with self.assertRaises(ValueError) as cm:
            output = layer(input)
        return

    def test_input_dim(self):
        input = torch.randint(255, size=(3, 256, 256), dtype=torch.float32)
        layer = Generator(downsampler_factory=Downsampler, upsampler_factory=Upsampler)
        with self.assertRaises(ValueError) as cm:
            output = layer(input)
        return


if __name__ == "__main__":
    unittest.main()
