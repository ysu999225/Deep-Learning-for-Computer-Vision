import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
from absl import flags, app
from utils import get_mnist_dataset, compute_accuracy, visualize_weights, visualize_knn
import logging
from tensorboardX import SummaryWriter
from featurize import featurize
from models import LinearClassifier, NearestNeighbor

FLAGS = flags.FLAGS
flags.DEFINE_enum('classifier', 'knn', ['knn', 'linear'], 'What classifier to run.')
flags.DEFINE_enum('feature', 'raw', ['raw', 'pool', 'hog'], 'What features to use.')
flags.DEFINE_string('out_dir', 'runs-linear-v2', 'Where to store results.')
flags.DEFINE_multi_integer('num_train', 1000, 'Number of training points')
flags.DEFINE_multi_integer('k', 1, 'k for kNN')
flags.DEFINE_multi_float('lr', 1e-1, 'Learning rates to run')
flags.DEFINE_multi_float('wt', 1e-3, 'Weight decay to use')
flags.DEFINE_integer('epochs', 10000, 'Number of iterations')
flags.DEFINE_integer('pool_size', 2, 'Pool size for pooling features')

def linear():
    data_val, labels_val = get_mnist_dataset('val')
    data_val = featurize(data_val, FLAGS.feature, pool_size=FLAGS.pool_size)
    
    for num_train in FLAGS.num_train:
        data_train, labels_train = get_mnist_dataset('train', num_train)
        data_train = featurize(data_train, FLAGS.feature, pool_size=FLAGS.pool_size)

        epochs = FLAGS.epochs
        for lr in FLAGS.lr:
            for wt in FLAGS.wt:
                dir_name = f'{FLAGS.out_dir}/num-{num_train}_lr-{lr}_wt-{wt}'
                writer = SummaryWriter(dir_name, flush_secs=30)
                linear = LinearClassifier(data_train, labels_train, epochs=epochs, 
                                          lr=lr, reg_wt=wt, writer=writer)
                linear.train()
                preds_val = linear.predict(data_val)
                accuracy = compute_accuracy(labels_val, preds_val)
                writer.add_scalar('val_accuracy', accuracy, epochs)
                logging.info(f'num_train: {num_train}, lr: {lr}, wt: {wt}, accuracy: {accuracy}')
                with open(dir_name + '/val_accuracy.txt', 'w') as f:
                    f.write(f'{accuracy}\n')
                if FLAGS.feature == 'raw':
                    # visualize the linear classifier weights
                    fig = visualize_weights(linear.w, file_name=f'{dir_name}/w_vis.png')
 
def knn():
    data_val, labels_val = get_mnist_dataset('val')
    pool_size = FLAGS.pool_size
    data_val = featurize(data_val, FLAGS.feature, pool_size=pool_size)
    
    for num_train in FLAGS.num_train:
        data_train, labels_train = get_mnist_dataset('train', num_train)
        data_train = featurize(data_train, FLAGS.feature, pool_size=pool_size)

        for k in FLAGS.k:
            dir_name = f'{FLAGS.out_dir}/num-{num_train}_k-{k}'
            writer = SummaryWriter(dir_name, flush_secs=30)
            knn = NearestNeighbor(data_train, labels_train, k)
            preds_val = knn.predict(data_val)
            accuracy = compute_accuracy(labels_val, preds_val)
            writer.add_scalar('val_accuracy', accuracy, 0)
            with open(dir_name + '/val_accuracy.txt', 'w') as f:
                f.write(f'{accuracy}\n')
            logging.info(f'num_train: {num_train}, k: {k}, accuracy: {accuracy}')


def main(_):
    if FLAGS.classifier == 'linear':
        linear()
    elif FLAGS.classifier == 'knn':
        knn()

if __name__ == '__main__':
    app.run(main)

