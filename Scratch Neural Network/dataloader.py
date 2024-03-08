import os
import gzip
import urllib.error
from urllib.request import urlretrieve

import numpy as np


class SimpleDataset:
    def __init__(self, split, batchsize=32):
        self.split = split
        self.batchsize = batchsize
        self.rng = np.random.RandomState(42)
        self.data = None
        self.labels = None
        self.batch_idx = 0

    def shuffle(self):
        perm = self.rng.permutation(len(self.data))
        self.data = self.data[perm]
        self.labels = self.labels[perm]

    def __len__(self):
        return len(self.data) // self.batchsize

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_idx >= len(self):
            self.shuffle()
            self.batch_idx = 0
            raise StopIteration

        start = self.batch_idx * self.batchsize
        end = start + self.batchsize
        self.batch_idx += 1
        return self.data[start:end], self.labels[start:end]


class CircleDataset(SimpleDataset):
    def __init__(self, split, batchsize=32):
        super().__init__(split, batchsize)
        if split == 'train':
            self.rng = np.random.RandomState(0)
        elif split == 'val':
            self.rng = np.random.RandomState(1)
        elif split == 'test':
            self.rng = np.random.RandomState(2)
        else:
            raise ValueError(f'Invalid split: {split}')

        self.data = self.rng.uniform(-10, 10, (1024, 2))
        self.labels = np.hstack([np.sum(self.data ** 2, axis=1, keepdims=True) < 50,
                                 np.sum(self.data ** 2, axis=1, keepdims=True) >= 50])
        self.labels = self.labels.astype(np.float32)

        self.batchsize = batchsize
        self.num_batches = len(self.data) // self.batchsize
        self.batch_idx = 0

    def num_classes(self):
        return 2

    def num_features(self):
        return 2


FILES = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz']


def download_mnist(path='data'):
    # base_url = 'http://yann.lecun.com/exdb/mnist/'
    base_url = 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/'
    for file in FILES:
        try:
            fn = os.path.join(path, file)
            if not os.path.exists(fn):
                print(f'Downloading {file} ...')
                urlretrieve(base_url + file, fn)
            else:
                # print(f'Found {file}, skipping ...')
                pass
        except urllib.error.URLError as e:
            print('Error downloading MNIST data:', e)


def load_mnist(path='data'):
    os.makedirs(path, exist_ok=True)
    download_mnist(path)
    data_path, label_path = FILES
    with gzip.open(os.path.join(path, data_path), 'rb') as f:
        imgs = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    with gzip.open(os.path.join(path, label_path), 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    return imgs, labels


class MNISTDataset(SimpleDataset):
    def __init__(self, split, batchsize=32):
        super().__init__(split, batchsize)
        data, labels = load_mnist('data')
        n1, n2, n3 = 40000, 10000, 10000

        if split == 'train':
            self.data = data[:n1, :, :].astype(float) / 255.0
            self.labels = labels[:n1]
        elif split == 'val':
            self.data = data[n1:n2 + n1, :, :].astype(float) / 255.0
            self.labels = labels[n1:n1 + n2]
        elif split == 'test':
            self.data = data[n1 + n2:n1 + n2 + n3, :, :].astype(float) / 255.0
            self.labels = labels[n1 + n2:n1 + n2 + n3]

        self.data = self.data[:, :, :, np.newaxis]
        self.data = self.data - 0.1307
        self.data = self.data / 0.3081
        self.labels = np.eye(10)[self.labels]

    @property
    def num_classes(self):
        return 10

    @property
    def num_features(self):
        return 28 * 28
