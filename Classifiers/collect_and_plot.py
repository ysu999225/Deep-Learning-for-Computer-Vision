import numpy as np
import glob
import matplotlib.pyplot as plt

def get_results(pattern, num_train):
    val_accs = []
    for num in num_train:
        pattern_num = pattern.replace('NUM', str(num))
        val_acc = np.loadtxt(f'{pattern_num}/val_accuracy.txt')
        val_accs.append(val_acc)
    return val_accs 

def plot_helper(datas, num_train, file_name):
    val_accs = []
    labels = []
    for pattern, label in datas:
        val_acc = get_results(pattern, num_train)
        val_accs.append(val_acc)
        labels.append(label)
    val_accs = np.array(val_accs)
    fig = plt.figure(figsize=(10, 10))
    fig.gca().semilogx(num_train, val_accs.T, '-o', label=labels)
    fig.gca().set_xlabel('Number of training examples')
    fig.gca().set_ylabel('Validation accuracy')
    fig.gca().set_title('Validation accuracy vs number of training examples')
    fig.gca().set_ylim([0.70, 1])
    fig.gca().grid(True)
    fig.gca().legend()
    plt.savefig(file_name, bbox_inches='tight',
                backend='agg')
    plt.close()

def plot_all():
    datas = [
        ('runs/linear-raw/num-NUM_lr-0.1_wt-0.001', 'linear (raw)'),
        ('runs/linear-pool2/num-NUM_lr-0.1_wt-0.001', 'linear (pool)'),
        ('runs/linear-hog4/num-NUM_lr-0.1_wt-0.001', 'linear (hog)'),
        ('runs/knn-raw/num-NUM_k-1', 'kNN (raw)'),
        ('runs/knn-pool2/num-NUM_k-1', 'kNN (pool)'),
        ('runs/knn-hog7/num-NUM_k-1', 'kNN (hog)'),
    ]
    num_train = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    
    plot_helper(datas[:3], num_train, 'performance_vs_num_train_linear.png')
    
    plot_helper(datas[3:], num_train, 'performance_vs_num_train_knn.png')

if __name__ == '__main__':
    plot_all()

