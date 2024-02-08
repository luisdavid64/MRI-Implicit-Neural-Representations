 # Current system path
import os
from pathlib import Path

# Get the current working directory
working_path = Path(os.getcwd()).joinpath(Path("src"))
import sys
sys.path.append(str(working_path))


import unittest
import os
import torch
import torch.nn as nn
from models.regularization import Regularization_L1, Regularization_L2

# Basic MLP
class SimpleLinearModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

class TestAddFunction(unittest.TestCase):
    def test_L1(self):
        # Test with random model parameters
        model = SimpleLinearModel(input_size=3, output_size=1)
        model_parameters_vector = torch.nn.utils.parameters_to_vector(model.parameters())

        # Instantiate Regularization_L1 with reg_strength=0.001
        l1_regularizer = Regularization_L1(reg_strength=0.001)
        
        # Calculate the expected L1 regularization value
        expected_result = torch.sum(torch.abs(model_parameters_vector)) * 0.001
        
        # Call the __call__ method and check if it produces the expected result
        result = l1_regularizer(model.parameters())
        
        # Assert that the result is close to the expected result
        self.assertEqual(result, expected_result)
        pass

    def test_L2(self):
        model = SimpleLinearModel(input_size=3, output_size=1)
        model_parameters_vector = torch.nn.utils.parameters_to_vector(model.parameters())

        # Instantiate Regularization_L1 with reg_strength=0.001
        l2_regularizer = Regularization_L2(reg_strength=0.001)
        
        # Calculate the expected L1 regularization value
        expected_result = torch.sum(model_parameters_vector.pow(2)) * 0.001
        
        # Call the __call__ method and check if it produces the expected result
        result = l2_regularizer(model.parameters())
        
        # Assert that the result is close to the expected result
        self.assertEqual(result, expected_result)
        pass

if __name__ == '__main__':
    unittest.main()