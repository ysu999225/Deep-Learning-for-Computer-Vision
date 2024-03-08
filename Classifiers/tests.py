import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import unittest
from utils_tests import load_variables
import numpy as np
from gradescope_utils.autograder_utils.decorators import \
    weight, visibility, number
from utils import get_mnist_dataset, compute_accuracy

class TestClassifier(unittest.TestCase):
    def setUp(self):
        pass

    def _test_knn(self, num_train, k, feature, pool_size, ref_accuracy,
                  delta=0.02, low=None, high=None):

        from featurize import featurize
        from models import NearestNeighbor
        data_val, labels_val = get_mnist_dataset('val')
        data_val = featurize(data_val, feature, pool_size=pool_size)

        data_train, labels_train = get_mnist_dataset('train', num_train)
        data_train = featurize(data_train, feature, pool_size=pool_size)

        nn = NearestNeighbor(data_train, labels_train, k=k)
        nn.train()
        preds_val = nn.predict(data_val)
        accuracy = compute_accuracy(labels_val, preds_val)
        if delta is not None:
            self.assertAlmostEqual(accuracy, ref_accuracy, delta=delta)
        else:
            self.assertGreaterEqual(accuracy, low)
            self.assertLessEqual(accuracy, high)

    def _test_linear(self, num_train, epochs, lr, wt, feature, pool_size,
                     ref_accuracy, delta=0.02):
        from featurize import featurize
        from models import LinearClassifier
        data_val, labels_val = get_mnist_dataset('val')
        data_val = featurize(data_val, feature, pool_size=pool_size)

        data_train, labels_train = get_mnist_dataset('train', num_train)
        data_train = featurize(data_train, feature, pool_size=pool_size)

        nn = LinearClassifier(data_train, labels_train, epochs=epochs, lr=lr,
                              reg_wt=wt)
        nn.train()
        preds_val = nn.predict(data_val)
        accuracy = compute_accuracy(labels_val, preds_val)
        
        self.assertAlmostEqual(accuracy, ref_accuracy, delta=delta)
    
    @weight(0.5)
    @number("1.1")
    @visibility('visible')
    def test_knn_small(self):
        self._test_knn(100, 5, 'raw', None, 0.6407, delta=None, low=0.6053, high=0.7121)
        self._test_knn(100, 15, 'raw', None, 0.5292, delta=None, low=0.5105, high=0.5869)

    @weight(1.5)
    @number("1.1")
    @visibility('visible')
    def test_knn(self):
        self._test_knn(1000, 5, 'raw', None, 0.8579, delta=None, low=0.8435, high=0.8762)
        self._test_knn(1000, 15, 'raw', None, 0.8249, delta=None, low=0.8143, high=0.8411)
    
    @weight(1.5)
    @number("3.1")
    @visibility('visible')
    def test_knn_pool(self):
        self._test_knn(1000, 1, 'pool', 4, 0.8514)
        self._test_knn(1000, 1, 'pool', 7, 0.6905)
        self._test_knn(1000, 1, 'pool', 14, 0.3908)
    
    @weight(1)
    @number("5.1")
    @visibility('visible')
    def test_knn_hog(self):
        self._test_knn(1000, 1, 'hog', 7, 0.8903, delta=0.02)

    @weight(0.5)
    @number("2.1")
    @visibility('visible')
    def test_linear_small(self):
        self._test_linear(100, 10000, 1e-1, 1e-3, 'raw', None, 0.7128)
        self._test_linear(100, 10000, 1e-1, 0, 'raw', None, 0.7150)

    @weight(1.5)
    @number("2.1")
    @visibility('visible')
    def test_linear(self):
        self._test_linear(1000, 10000, 1e-1, 1e-3, 'raw', None, 0.8604)
        self._test_linear(1000, 10000, 1e-1, 1, 'raw', None, 0.6911)
        self._test_linear(1000, 10000, 1e-4, 1e-3, 'raw', None, 0.6688)
    
    @weight(1.5)
    @number("3.2")
    @visibility('visible')
    def test_linear_pool(self):
        self._test_linear(1000, 10000, 1e-1, 1e-3, 'pool', 2, 0.8654)
        self._test_linear(1000, 10000, 1e-1, 1e-3, 'pool', 4, 0.8321)
        self._test_linear(1000, 10000, 1e-1, 1e-3, 'pool', 14, 0.1536)
    
    @weight(1)
    @number("5.2")
    @visibility('visible')
    def test_linear_hog(self):
        self._test_linear(1000, 10000, 1e-1, 1e-3, 'hog', 4, 0.9396, delta=0.02)

    @weight(2)
    @number("2.1")
    @visibility('visible')
    def test_gradient_and_loss(self):
        from models import LinearClassifier
        samples = load_variables('test-data/linear.pkl')['samples']
        for i, sample in enumerate(samples):
            linear = LinearClassifier(sample['data'], sample['labels'],
                                      epochs=1, lr=1e-1, reg_wt=1e-3)
            linear.w = sample['w']

            data_loss, reg_loss, total_loss, grad_w = \
                linear.compute_loss_and_gradient()
            
            self.assertAlmostEqual(data_loss, sample['data_loss'])
            self.assertAlmostEqual(reg_loss, sample['reg_loss'])
            self.assertAlmostEqual(total_loss, sample['total_loss'])
            self.assertTrue(np.allclose(grad_w, sample['grad_w'], rtol=1e-3, atol=1e-5))

if __name__ == '__main__':
    unittest.main()
