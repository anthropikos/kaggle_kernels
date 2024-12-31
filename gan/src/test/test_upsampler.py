## Anthony Lee 2024-12-29

import unittest
import torch
from gan.upsampler import Upsampler


class Test_Upsampler(unittest.TestCase):
    def test_defaultUpsampler(self):
        """Verify that the default upsampler halves the width and height."""

        batch_size, channels, height, width = 5, 3, 128, 128

        input = torch.randint(255, (batch_size, channels, height, width)).to(dtype=torch.float32)
        layer = Upsampler(filters=channels)
        output = layer(input)

        self.assertEqual(output.size(), (batch_size, channels, height * 2, width * 2))
        return

    def test_float64_input(self):
        """The model should raise an error when providing non float32 type."""
        input = torch.randint(255, (5, 3, 256, 256), dtype=torch.float64)
        layer = Upsampler(filters=3)
        with self.assertRaises(RuntimeError) as cm:
            output = layer(input)
        return

    def test_nonbatched_input(self):
        """The model only accept batched input, thus intput has to have 4-dimensions."""
        input = torch.randint(255, (3, 256, 256), dtype=torch.float32)
        layer = Upsampler(filters=3)
        with self.assertRaises(ValueError) as cm:
            output = layer(input)

        return


if __name__ == "__main__":
    unittest.main()
