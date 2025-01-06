## Anthony Lee 2024-12-29

import unittest
import torch
from gan.downsampler import Downsampler


class Test_Downsampler(unittest.TestCase):

    def test_float64_input(self):
        n_filters = 3
        input = torch.randint(255, (5, 3, 256, 256), dtype=torch.float64)
        layer = Downsampler(filters=n_filters)

        # Method 1
        # unittest.TestCase.assertRaises(self, RuntimeError, layer, input)

        # Method 2
        with self.assertRaises(RuntimeError) as cm:
            output = layer(input)
        return

    def test_simple(self):
        self.assertEqual(4, 4)
        return

    def test_nonbatched_input(self):
        input = torch.randint(255, (3, 256, 256), dtype=torch.float32)
        layer = Downsampler(filters=3)
        with self.assertRaises(ValueError) as cm:
            output = layer(input)
        return


class Test_Downsampler(unittest.TestCase):
    def test_defaultDownsampler(self):
        """Verify that the default downsampler halves the width and height."""

        batch_size, channels, height, width = 5, 3, 256, 256

        input = torch.randint(255, (batch_size, channels, height, width)).to(dtype=torch.float32)
        layer = Downsampler(filters=channels)
        output = layer(input)

        self.assertEqual(output.size(), (batch_size, channels, height / 2, width / 2))
        return

    def test_padding_1(self):
        batch_size, channels, height, width = 5, 3, 256, 256
        padding = 1
        input = torch.randint(low=0, high=10, size=(batch_size, channels, height, width), dtype=torch.float32)
        layer = Downsampler(filters=1, padding=padding)
        output = layer(input)

        self.assertEqual(output.size(), torch.Size((5, 1, 128, 128)))
        return

    def test_padding_2(self):
        batch_size, channels, height, width = 5, 3, 256, 256
        padding = 2
        input = torch.randint(low=0, high=10, size=(batch_size, channels, height, width), dtype=torch.float32)
        layer = Downsampler(filters=1, padding=padding)
        output = layer(input)

        self.assertEqual(output.size(), torch.Size((5, 1, 129, 129)))
        return


if __name__ == "__main__":
    unittest.main()
