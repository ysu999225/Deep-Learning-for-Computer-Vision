import os
import time

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # export NUMEXPR_NUM_THREADS=6

import numpy as np
from absl import app, flags
from tensorboardX import SummaryWriter
from tqdm import tqdm

from dataloader import CircleDataset, MNISTDataset
from layers import Linear, ReLU, Sequential, Flatten, Conv2d, MaxPool2d
from losses import L2Loss, SoftmaxWithLogitsLoss
from optimizer import SGD

FLAGS = flags.FLAGS
flags.DEFINE_enum('classifier', 'linear', ['linear', 'nn', 'cnn'], 'What classifier to run.')
flags.DEFINE_string('root', 'data', 'Data root path')
flags.DEFINE_integer('num_train', 1000, 'Number of training points')
flags.DEFINE_float('lr', 1e-2, 'Learning rates to run')
flags.DEFINE_float('gamma', 1., 'Learning rates decay')
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_integer('epochs', 100, 'Number of iterations')


def val_loop(dataloader, net, loss_fn):
    total_loss = 0
    total_correct = 0
    n_batches = len(dataloader)

    for x, y in tqdm(dataloader, desc='Validation'):
        o = net.forward(x)
        loss = loss_fn.forward(o, y)
        total_loss += loss
        total_correct += np.sum(np.argmax(o, axis=1) == np.argmax(y, axis=1))

    total_loss /= n_batches
    accuracy = total_correct / (n_batches * x.shape[0])
    return total_loss, accuracy


def train_loop(dataloader, net, loss_fn, optim, lr, epoch):
    total_loss = 0
    total_correct = 0
    n_batches = len(dataloader)

    for i, (x, y) in enumerate(dataloader):
        o = net.forward(x)
        loss = loss_fn.forward(o, y)
        delta = loss_fn.backward(1)
        net.backward(delta)
        optim.step(lr)
        total_loss += loss
        total_correct += np.sum(np.argmax(o, axis=1) == np.argmax(y, axis=1))

        if i % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i, n_batches, 100. * i / n_batches, loss.item()))

    total_loss /= n_batches
    accuracy = total_correct / (n_batches * x.shape[0])
    return total_loss, accuracy


def setup_linear_mnist():
    return Sequential(Flatten(), Linear(28 * 28, 10))


def setup_nn_mnist():
    return Sequential(
        Flatten(),
        Linear(28 * 28, 128),
        ReLU(),
        Linear(128, 10)
    )


def setup_cnn_mnist():
    return Sequential(
        Conv2d(1, 32, [3, 3]),
        ReLU(),
        Conv2d(32, 64, [3, 3]),
        ReLU(),
        MaxPool2d([2, 2]),
        Flatten(),
        Linear(9216, 128),
        ReLU(),
        Linear(128, 10)
    )


def setup_circle():
    trainloader = CircleDataset('train', 32)
    valloader = CircleDataset('val', 32)
    testloader = CircleDataset('test', 32)

    writer = SummaryWriter('runs/circle-dataset-demo')

    net = Sequential(Linear(trainloader.num_features(), trainloader.num_classes()))

    return (trainloader, valloader, testloader), writer, net


def main(_):
    start = time.time()
    trainloader = MNISTDataset('train', 32)
    valloader = MNISTDataset('val', 32)
    testloader = MNISTDataset('test', 32)
    loss_fn = SoftmaxWithLogitsLoss()
    writer = SummaryWriter(f'runs/mnist-{FLAGS.classifier}-demo-lr{FLAGS.lr}_gamma{FLAGS.gamma}_epochs{FLAGS.epochs}')
    if FLAGS.classifier == 'linear':
        net = setup_linear_mnist()
    elif FLAGS.classifier == 'nn':
        net = setup_nn_mnist()
    elif FLAGS.classifier == 'cnn':
        net = setup_cnn_mnist()
    else:
        raise NotImplementedError
    net.initialize(0)
    optim = SGD(net)
    lr = FLAGS.lr

    for epoch in range(FLAGS.epochs):
        train_loss, train_accuracy = train_loop(trainloader, net, loss_fn, optim, lr, epoch)
        writer.add_scalar('train_loss', train_loss, epoch + 1)
        writer.add_scalar('train_accuracy', train_accuracy, epoch + 1)

        val_loss, val_accuracy = val_loop(valloader, net, loss_fn)
        writer.add_scalar('val_loss', val_loss, epoch + 1)
        writer.add_scalar('val_accuracy', val_accuracy, epoch + 1)

        if epoch % 1 == 0:
            print(f'\tEpoch: {epoch + 1}, Train Loss: {train_loss:5.3f}, Train Accuracy: {train_accuracy:5.3f}')
            print(f'\tEpoch: {epoch + 1}, Val Loss: {val_loss:5.3f}, Val Accuracy: {val_accuracy:5.3f}')
        lr = lr * FLAGS.gamma

    test_loss, test_accuracy = val_loop(valloader, net, loss_fn)
    print(f'Test Loss: {test_loss:5.3f}, Test Accuracy: {test_accuracy:5.3f}')
    took = time.time() - start
    print(f'Time taken: {int(took // 60):d}m {took % 60:.2f}s')


if __name__ == '__main__':
    app.run(main)
