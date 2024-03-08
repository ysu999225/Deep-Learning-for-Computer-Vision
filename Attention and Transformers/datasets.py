import torchvision as tv
import numpy as np

def get_transforms(split, size):
    normalize = tv.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    if size == 224:
        resize_dim = 256
        crop_dim = 224
    if split == "train":
        transform = tv.transforms.Compose(
            [
                tv.transforms.Resize(resize_dim, antialias=True),
                tv.transforms.RandomCrop(crop_dim),
                tv.transforms.RandomHorizontalFlip(0.5),
                tv.transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        transform = tv.transforms.Compose(
            [
                tv.transforms.Resize(resize_dim, antialias=True),
                tv.transforms.CenterCrop(crop_dim),
                tv.transforms.ToTensor(),
                normalize,
            ]
        )
    return transform

def get_cifar(split):
    rng = np.random.RandomState(1)
    transform = get_transforms(split, 224)
    dataset = tv.datasets.CIFAR100('datasets/', train=split in ['train', 'val'],
                                   download=False, transform=transform)
    ind = rng.permutation(len(dataset.data))
    if split == 'train':
        ind = ind[:800]
    elif split == 'val':
        ind = ind[800:1200]
    elif split == 'test':
        import pickle
        with open('./datasets/cifar-100-python/test_mp4', "rb") as f:
            entry = pickle.load(f, encoding="latin1")
        dataset.data, dataset.targets = entry['data'], entry['fine_labels']
        return dataset
    
    dataset.data = dataset.data[ind, ...]
    dataset.targets = [dataset.targets[x] for x in ind]
    return dataset