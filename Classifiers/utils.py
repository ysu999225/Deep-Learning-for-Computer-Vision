import os
import gzip
import urllib.error
from urllib.request import urlretrieve

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines

# Following code for loading mnist, from pytorch repo.
FILES = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz']

def download_mnist(path='data'):
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    for file in FILES:
        try:
            fn = os.path.join(path, file)
            if not os.path.exists(fn):
                print(f'Downloading {file} ...')
                urlretrieve(base_url + file, fn)
            else:
                # print(f'Found {file}, skipping ...')
                None
        except urllib.error.URLError as e:
            print('Error downloading MNIST data:', e)


def load_mnist(path='data'):
    os.makedirs(path, exist_ok=True)
    download_mnist(path)
    data_path, label_path = FILES
    with gzip.open(os.path.join(path, data_path), 'rb') as f:
        imgs = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)
    with gzip.open(os.path.join(path, label_path), 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    return imgs, labels


def get_mnist_dataset(split='train', size=40000, seed=0):
    data, labels = load_mnist('data')
    n1, n2, n3 = 40000, 10000, 10000

    rng = np.random.RandomState(seed)
    ind = rng.permutation(data.shape[0])
    data = data[ind]
    labels = labels[ind]
    
    if split == 'train':
        data = data[:n1, :].astype(float) / 255.0
        labels = labels[:n1]
    elif split == 'val':
        data = data[n1:n2 + n1, :].astype(float) / 255.0
        labels = labels[n1:n1 + n2]
    elif split == 'test':
        data = data[n1 + n2:n1 + n2 + n3, :].astype(float) / 255.0
        labels = labels[n1 + n2:n1 + n2 + n3]
    return data[:size], labels[:size]


def compute_accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()


def visualize_knn(x, knn_x, file_name=None):
    n, k = x.shape[0], knn_x.shape[1]
    fig, axes = plt.subplots(n, k + 1, figsize=(10, 10), sharex=True, sharey=True,
                             gridspec_kw=dict(hspace=-0.01, wspace=-0.01))
    for i in range(n):
        axes[i, 0].imshow(x[i].reshape(28, 28), cmap='gray')
        for j in range(k):
            axes[i, j + 1].imshow(knn_x[i, j].reshape(28, 28), cmap='gray')
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # Draw a line separating the query image from the neighbors
    xline = fig.transFigure.inverted().transform(
        axes[0, 0].transAxes.transform([1, 1])
    )[0]
    fig.add_artist(lines.Line2D([xline, xline], [0, 1], color='white'))
    fig.text(0.5, 0.9, 'Query image (left) & nearest neighbors', ha='center', fontsize=24)
    if file_name is not None:
        plt.savefig(file_name, bbox_inches=None, backend='Agg')
        plt.close()
    else:
        plt.show()

def visualize_weights(weights, file_name=None):
    weights = weights.reshape((28, 28, 10)).transpose((2, 0, 1))
    fig, axes = plt.subplots(5, 2, figsize=(2, 5), sharex=True, sharey=True,
                             gridspec_kw=dict(hspace=-0.01, wspace=-0.01))
    for ax, img in zip(axes.flat, weights):
        ax.imshow(img, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(0.5, 0.9, 'Classifier weights', ha='center', fontsize=16)
    if file_name is not None:
        plt.savefig(file_name, bbox_inches=None, backend='Agg')
        plt.close()
    else:
        plt.show()

