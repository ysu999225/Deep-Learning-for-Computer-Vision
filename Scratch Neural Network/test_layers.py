import unittest

import numpy as np
from gradescope_utils.autograder_utils.decorators import weight, visibility, number

import utils
from layers import Linear, Sequential, ReLU, Conv2d, MaxPool2d
from losses import SoftmaxWithLogitsLoss

conv_data_file = './test_data/conv.pkl'
relu_data_file = './test_data/relu.pkl'
sequential_data_file = './test_data/sequential.pkl'
maxpool_data_file = './test_data/maxpool.pkl'
softmax_with_logits_data_file = './test_data/softmax_with_logits_loss.pkl'


class TestLayersAndLosses(unittest.TestCase):
    def setUp(self) -> None:
        pass

    #######################################
    # Test Losses                         #
    #######################################

    def _test_loss(self, loss, test_data_file,
                   forward=True, grad_w=True, grad_x=True,
                   rtol=1e-5, atol=1e-8):
        dt = utils.load_variables(test_data_file)
        for i in range(len(dt['samples'])):
            sample = dt['samples'][i]
            l = loss(*sample['layer_params'])
            x_out = l.forward(sample['x_in'], sample['y_in'])

            if forward:
                self.assertTrue(
                    x_out.shape == sample['x_out'].shape,
                    f'Shape of loss {x_out.shape} does not match expected shape {sample["x_out"].shape}!'
                )
                self.assertTrue(
                    np.allclose(x_out, sample['x_out'], rtol=rtol, atol=atol),
                    f'Output loss {x_out:.4f} does not match expected loss {sample["x_out"]:.4f}!'
                )

            if grad_w or grad_x:
                x_in_grad = l.backward(sample['delta_out'])
                if grad_x:
                    self.assertTrue(
                        x_in_grad.shape == sample['grad_x_in'].shape,
                        f'Shape of gradient {x_in_grad.shape} does not match expected'
                        f' shape {sample["grad_x_in"].shape}!'
                    )
                    self.assertTrue(
                        np.allclose(x_in_grad, sample['grad_x_in'], atol=atol, rtol=rtol),
                        f'Gradient of loss does not match expected gradient!'
                    )

    @weight(0.5)
    @number("1.3.1")
    @visibility('visible')
    def test_loss_softmax_with_logits_forward(self):
        self._test_loss(SoftmaxWithLogitsLoss, softmax_with_logits_data_file, True, False, False)

    @weight(0.5)
    @number("1.3.2")
    @visibility('visible')
    def test_loss_softmax_with_logits_grad_x(self):
        self._test_loss(SoftmaxWithLogitsLoss, softmax_with_logits_data_file, False, False, True)

    #######################################
    # Test Layers                         #
    #######################################

    def _test_layer(self, layer, test_data_file,
                    forward=True, grad_w=True, grad_x=True,
                    rtol=1e-5, atol=1e-8):
        dt = utils.load_variables(test_data_file)
        for i in range(len(dt['samples'])):
            sample = dt['samples'][i]
            l = layer(*sample['layer_params'])

            for k in l.params.keys():
                l.params[k] = sample[k]
            x_out = l.forward(sample['x_in'])

            if forward:
                self.assertTrue(
                    x_out.shape == sample['x_out'].shape,
                    f'Shape of output {x_out.shape} does not match expected'
                    f' shape {sample["x_out"].shape} for `{layer.__name__}`!'
                )
                self.assertTrue(
                    np.allclose(x_out, sample['x_out'], rtol=rtol, atol=atol),
                    f'Output does not match expected output for `{layer.__name__}`!'
                )

            if grad_w or grad_x:
                x_in_grad = l.backward(sample['delta_out'])
                if grad_w:
                    for k in l.params.keys():
                        self.assertTrue(
                            l.grads[k].shape == sample['grad_' + k].shape,
                            f'Shape of gradient {l.grads[k].shape} does not match expected'
                            f' shape {sample["grad_" + k].shape} for `{layer.__name__}`!'
                        )
                        self.assertTrue(
                            np.allclose(l.grads[k], sample['grad_' + k], rtol=rtol, atol=atol),
                            f'Gradient wrt to weights does not match expected gradient for `{layer.__name__}`!'
                        )
                if grad_x:
                    self.assertTrue(
                        x_in_grad.shape == sample['grad_x_in'].shape,
                        f'Shape of gradient {x_in_grad.shape} does not match expected'
                        f' shape {sample["grad_x_in"].shape} for `{layer.__name__}`!'
                    )
                    self.assertTrue(
                        np.allclose(x_in_grad, sample['grad_x_in'], atol=atol, rtol=rtol),
                        f'Gradient wrt to input does not match expected gradient for `{layer.__name__}`!'
                    )

    @weight(1)
    @number("1.4")
    @visibility('visible')
    def test_sequential(self):
        rtol, atol = 1e-5, 1e-8
        dt = utils.load_variables(sequential_data_file)
        for i in range(len(dt['samples'])):
            sample = dt['samples'][i]
            in_d, out_d = sample['layer_params']
            l = Sequential(
                Linear(in_d, out_d),
                ReLU(),
                Linear(out_d, out_d)
            )
            l.layers[0].params['weight'] = sample['w0']
            l.layers[0].params['bias'] = sample['b0']
            l.layers[2].params['weight'] = sample['w1']
            l.layers[2].params['bias'] = sample['b1']
            x_out = l.forward(sample['x_in'])

            self.assertTrue(
                x_out.shape == sample['x_out'].shape,
                f'Shape of output {x_out.shape} does not match expected'
                f' shape {sample["x_out"].shape} for `Sequential`!'
            )
            self.assertTrue(
                np.allclose(x_out, sample['x_out'], rtol=rtol, atol=atol),
                f'Output does not match expected output for `Sequential`!'
            )

            x_in_grad = l.backward(sample['delta_out'])
            self.assertTrue(
                x_in_grad.shape == sample['grad_x_in'].shape,
                f'Shape of gradient {x_in_grad.shape} does not match expected'
                f' shape {sample["grad_x_in"].shape} for `Sequential`!'
            )
            self.assertTrue(
                np.allclose(x_in_grad, sample['grad_x_in'], atol=atol, rtol=rtol),
                f'Gradient wrt to input does not match expected gradient for `Sequential`!'
            )

    @weight(1.5)
    @number("2.1.1")
    @visibility('visible')
    def test_layer_maxpool_forward(self):
        self._test_layer(MaxPool2d, maxpool_data_file, True, False, False)

    @weight(1.5)
    @number("2.1.2")
    @visibility('visible')
    def test_layer_maxpool_grad_input(self):
        self._test_layer(MaxPool2d, maxpool_data_file, False, False, True)

    @weight(0.5)
    @number("1.1.1")
    @visibility('visible')
    def test_layer_relu_forward(self):
        self._test_layer(ReLU, relu_data_file, True, False, False)

    @weight(0.5)
    @number("1.1.2")
    @visibility('visible')
    def test_layer_relu_grad_input(self):
        self._test_layer(ReLU, relu_data_file, False, False, True)

    @weight(1)
    @number("2.2.1")
    @visibility('visible')
    def test_layer_conv2d_forward(self):
        self._test_layer(Conv2d, conv_data_file, True, False, False, 1e-5, 1e-4)

    @weight(1.5)
    @number("2.2.2")
    @visibility('visible')
    def test_layer_conv2d_grad_param(self):
        self._test_layer(Conv2d, conv_data_file, False, True, False, 1e-5, 1e-4)

    @weight(1.5)
    @number("2.2.3")
    @visibility('visible')
    def test_layer_conv2d_grad_input(self):
        self._test_layer(Conv2d, conv_data_file, False, False, True, 1e-5, 1e-4)


if __name__ == '__main__':
    unittest.main()
