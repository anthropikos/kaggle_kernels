import unittest
import gan
import gan.generator
import torch


class TestDownsampler(unittest.TestCase):

    def test_default_float32(self):
        n_filters=3
        input = torch.randint(255, (3, 256, 256)).double()
        layer = gan.generator.Downsampler(filters=n_filters)

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
        layer = gan.generator.Downsampler(filters=channels)
        output = layer(input)

        self.assertEqual(output.size(), (channels, height/2, width/2))

class Test_Upsampler(unittest.TestCase):
    def test_defaultUpsampler(self):
        """Verify that the default upsampler halves the width and height."""

        channels, height, width = 3, 128, 128

        input = torch.randint(255, (channels, height, width)).to(dtype=torch.float32)
        layer = gan.generator.Upsampler(filters=channels)
        output = layer(input)

        self.assertEqual(output.size(), (channels, height*2, width*2))


class Test_Generator(unittest.TestCase):
    def test_inputShape_unbatched_greater(self):
        """Test unbatched input when dimension is greater than 256."""
        input = torch.randint(255, size=(3, 257, 257)).float()
        layer = gan.generator.Generator(Downsampler=gan.generator.Downsampler, Upsampler=gan.generator.Upsampler)
        with self.assertRaises(AttributeError) as cm:
            output = layer(input)
        return

    def test_inputShape_unbatched_lesser(self):
        """Test unbatched input when dimension is lesser than 256."""
        input = torch.randint(255, size=(3, 255, 255)).float()
        layer = gan.generator.Generator(Downsampler=gan.generator.Downsampler, Upsampler=gan.generator.Upsampler)
        with self.assertRaises(AttributeError) as cm:
            output = layer(input)
        return
        
    def test_inputShape_batched_greater(self):
        """Test batched input when dimension is greater than 256."""
        input = torch.randint(255, size=(10, 3, 257, 257)).float()
        layer = gan.generator.Generator(Downsampler=gan.generator.Downsampler, Upsampler=gan.generator.Upsampler)
        with self.assertRaises(AttributeError) as cm:
            output = layer(input)
        return
    
    def test_inputShape_batched_lesser(self):
        """Test batched input when dimension is lesser than 256."""
        input = torch.randint(255, size=(10, 3, 255, 255)).float()
        layer = gan.generator.Generator(Downsampler=gan.generator.Downsampler, Upsampler=gan.generator.Upsampler)
        with self.assertRaises(AttributeError) as cm:
            output = layer(input)
        return

if __name__ == "__main__":
    unittest.main()