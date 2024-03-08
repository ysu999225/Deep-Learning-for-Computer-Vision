import unittest

import numpy as np
from gradescope_utils.autograder_utils.decorators import weight, visibility, number
from transformer_utils import SelfAttention, LayerNorm
import utils
import torch

test_self_attention_file = './test_data/self_attention.pkl'
test_layer_norm_file = './test_data/layer_norm.pkl'

class TestClass(unittest.TestCase):
    def setUp(self) -> None:
        pass

    ####################
    # Self Attention   #
    ####################

    def _test_self_attention(self, SelfAttention, test_data_file, rtol=1e-3, atol=1e-5):
        dt = utils.load_variables(test_data_file)
        torch.manual_seed(0)

        for i in range(len(dt)):
            input = dt[i]['input']
            x, weight = input['x'], input['weight']
            target = dt[i]['target']
            
            self_attention = SelfAttention(input_dim=x.size(-1), query_dim=64, 
                                        key_dim=64, value_dim=96)
            self_attention.load_state_dict(weight)

            attention = self_attention(x).detach()

            self.assertTrue(
                attention.shape == target.shape,
                f"Shape of calculated self attention is {attention.shape}, which does not match expected shape {target.shape}!"
            )

            self.assertTrue(
                np.allclose(attention, target, rtol=rtol, atol=atol),
                f'Value of calculated self attention does not match expected value!'
            )

    @weight(2.0)
    @number("1.1")
    @visibility('visible')
    def test_self_attention(self):
        self._test_self_attention(SelfAttention, test_self_attention_file)


    ###################
    # Test Layer Norm #
    ###################

    def _test_layer_norm(self, LayerNorm, test_data_file, rtol=1e-5, atol=1e-6):
        dt = utils.load_variables(test_data_file)

        for i in range(len(dt)):
            input = dt[i]['input']
            x, w, b = input['x'], input['w'], input['b']
            target = dt[i]['target']
            norm = LayerNorm(input_dim=x.size(-1))
            norm.w = torch.nn.Parameter(w)
            norm.b = torch.nn.Parameter(b)

            normalized = norm(x).detach()

            self.assertTrue(
                normalized.shape == target.shape,
                f"Shape of output from layer normalization is {normalized.shape}, which does not match expected shape {target.shape}!"
            )

            self.assertTrue(
                np.allclose(normalized, target, rtol=rtol, atol=atol),
                f'Value of output from layer normalization does not match expected value!'
            )

    @weight(2.0)
    @number("1.2")
    @visibility('visible')
    def test_layer_norm(self):
        self._test_layer_norm(LayerNorm, test_layer_norm_file)

if __name__ == '__main__':
    unittest.main()
