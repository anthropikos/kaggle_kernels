## Anthony Lee 2024-12-29

import unittest
import torch
from gan.downsampler import Downsampler

class TestDownsampler(unittest.TestCase):

    def test_default_float32(self):
        n_filters=3
        input = torch.randint(255, (3, 256, 256)).double()
        layer = Downsampler(filters=n_filters)

        # Method 1
        # unittest.TestCase.assertRaises(self, RuntimeError, layer, input)

        # Method 2
        with self.assertRaises(RuntimeError) as cm:
            output = layer(input)

    def test_simple(self):
        self.assertEqual(4, 4)


class Test_Downsampler(unittest.TestCase):
    def test_defaultDownsampler(self):
        """Verify that the default downsampler halves the width and height."""

        channels, height, width = 3, 256, 256

        input = torch.randint(255, (channels, height, width)).to(dtype=torch.float32)
        layer = Downsampler(filters=channels)
        output = layer(input)

        self.assertEqual(output.size(), (channels, height/2, width/2))

    def test_padding_1(self):
        batch_size, channels, height, width = 5, 3, 256, 256
        padding = 1
        input = torch.randint(low=0, high=10, size=(batch_size, channels, height, width), dtype=torch.float32)
        layer = Downsampler(filters=1, padding=padding)
        output = layer(input)

        self.assertEqual(output.size(), torch.Size((5, 1, 128, 128)))

    def test_padding_2(self):
        batch_size, channels, height, width = 5, 3, 256, 256
        padding = 2
        input = torch.randint(low=0, high=10, size=(batch_size, channels, height, width), dtype=torch.float32)
        layer = Downsampler(filters=1, padding=padding)
        output = layer(input)

        self.assertEqual(output.size(), torch.Size((5, 1, 129, 129)))


if __name__ == "__main__":
    unittest.main()