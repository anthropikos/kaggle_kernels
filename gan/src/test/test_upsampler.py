## Anthony Lee 2024-12-29

import unittest
import torch
from gan.upsampler import Upsampler

class Test_Upsampler(unittest.TestCase):
    def test_defaultUpsampler(self):
        """Verify that the default upsampler halves the width and height."""

        channels, height, width = 3, 128, 128

        input = torch.randint(255, (channels, height, width)).to(dtype=torch.float32)
        layer = Upsampler(filters=channels)
        output = layer(input)

        self.assertEqual(output.size(), (channels, height*2, width*2))

if __name__ == "__main__":
    unittest.main()