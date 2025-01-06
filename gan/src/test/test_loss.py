# Anthony Lee 2024-12-30

import unittest
import torch
import gan.loss as gl


class Test_DiscriminatorLoss(unittest.TestCase):
    def test_scalar_output(self):
        loss_func = gl.DiscriminatorLoss()
        input_size = (5, 3, 256, 256)
        real_tensor = torch.randint(255, size=input_size, dtype=torch.float32)
        generated_tensor = torch.randint(255, size=input_size, dtype=torch.float32)

        loss = loss_func(real_tensor, generated_tensor)
        self.assertEqual(loss.size(), torch.Size([]), msg="The loss func output has to be a scalar.")


class Test_GeneratorLoss(unittest.TestCase):
    def test_scalar_output(self):
        loss_func = gl.GeneratorLoss()
        input_size = (5, 3, 256, 256)
        generated_tensor = torch.randint(255, size=input_size, dtype=torch.float32)

        loss = loss_func(generated_tensor)
        self.assertEqual(loss.size(), torch.Size([]), msg="The loss func output has to be a scalar.")


class Test_CycleLoss(unittest.TestCase):
    def test_scalar_output(self):
        loss_func = gl.CycleLoss()
        input_size = (5, 3, 256, 256)
        scale = 10
        real_tensor = torch.randint(255, size=input_size, dtype=torch.float32)
        cycled_tensor = torch.randint(255, size=input_size, dtype=torch.float32)

        loss = loss_func(real=real_tensor, cycled=cycled_tensor, scale=scale)
        self.assertEqual(loss.size(), torch.Size([]), msg="The loss func output has to be a scalar.")


class Test_IdentityLoss(unittest.TestCase):
    def test_scalar_output(self):
        loss_func = gl.IdentityLoss()
        input_size = (5, 3, 256, 256)
        scale = 10
        input1 = torch.randint(255, size=input_size, dtype=torch.float32)
        input2 = torch.randint(255, size=input_size, dtype=torch.float32)

        loss = loss_func(real_image=input1, same_image=input2, scale=scale)
        self.assertEqual(loss.size(), torch.Size([]), msg="The loss func output has to be a scalar.")


if __name__ == "__main__":
    unittest.main()
