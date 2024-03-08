from torchvision.datasets import CocoDetection
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
from torch.utils.data import Dataset
from PIL import ImageOps
import numpy as np
import json
import torchvision as tv
from torchvision.transforms import v2
from torch.utils.data.sampler import Sampler
import random

import skimage.io
import skimage.transform
import skimage.color
import skimage

class CocoDataset(Dataset):
    def __init__(self, split='train', min_sizes=[800], 
                 seed=0, transform=None):
        self.annotation_file = '../coco/annotations/' + \
            f'instances_{split}2017-animal-nocrowd.json'
        self.coco = CocoDetection(f'../coco/{split}2017/',
                                  self.annotation_file)
        self.image_ids = self.coco.ids
        
        # Remap categories
        dt = json.load(open(self.annotation_file))
        cat_map = np.zeros(100, dtype=np.int64)
        classes = {}
        for i, cat in enumerate(dt['categories']):
            cat_map[cat['id']] = i + 1
            classes[i+1] = cat['name']
        self.cat_map = cat_map
        self.classes = classes
        self.num_classes = len(dt['categories'])

        self.rng = np.random.RandomState(seed=seed)
        self.min_sizes = min_sizes
        self.transform = transform

    def __len__(self):
        return len(self.coco)
    
    def _get_annotation(self, index):
        image, annotation = self.coco[index]
        # remap categories to 1 through 10 index
        # only return the bounding box
        # resize image appropriately
        cls = np.array([a['category_id'] for a in annotation]).astype(np.int64)
        cls = self.cat_map[cls]
        is_crowd = np.array([a['iscrowd'] for a in annotation])
        bboxes = np.array([a['bbox'] for a in annotation])
        image_id = [a['image_id'] for a in annotation][0]
        resize_factor = None
        
        valid_idx = [False if b[2] < 1 or b[3] < 1 else True for b in bboxes]
        anno_bboxes = bboxes[valid_idx]
        cls = cls[valid_idx]
        is_crowd = is_crowd[valid_idx]

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        anno_bboxes[:, 2] = anno_bboxes[:, 0] + anno_bboxes[:, 2]
        anno_bboxes[:, 3] = anno_bboxes[:, 1] + anno_bboxes[:, 3]

        if self.transform:
            image = np.asarray(image, dtype=np.float32) / 255
            image, anno_bboxes, cls, is_crowd, image_id, resize_factor = self.transform([image, anno_bboxes, cls[..., np.newaxis], is_crowd, image_id])


        return image, anno_bboxes, cls, is_crowd, image_id, resize_factor

    def __getitem__(self, index):
        """
        Args:
            index: int, index of the image
        Returns:
            image: (3, H, W) tensor, normalized to [0,1], mean subtracted, std divided
            cls: (N) tensor of class indices
            bbox: (N, 4) tensor of bounding boxes in the format (x1, y1, x2, y2)
            is_crowd: (N) tensor of booleans indicating whether the bounding box is a crowd
        """
        image, bboxes, cls, is_crowd, image_id, resize_factor = self._get_annotation(index)

        bboxes = bboxes[is_crowd == 0, :]
        cls = cls[is_crowd == 0]
        is_crowd = is_crowd[is_crowd == 0]
        
        return image, bboxes, cls, is_crowd, image_id, resize_factor

    def evaluate(self, result_file_name):
        coco_gt = COCO(self.annotation_file)
        coco_results = coco_gt.loadRes(result_file_name)
        cocoEval = COCOeval(coco_gt, coco_results, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        metrics = [cocoEval.stats]
        for catId in coco_gt.getCatIds():
            coco_eval = COCOeval(coco_gt, coco_results, 'bbox')
            coco_eval.params.catIds = [catId]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            metrics.append(coco_eval.stats)
        metrics = np.array(metrics) 
        return metrics, ['all'] + list(self.classes.values())
    
    def image_aspect_ratio(self, image_index):
        t = self.coco.coco.imgs[self.image_ids[image_index]]
        width, height = t['width'], t['height']
        return float(width) / float(height)

class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, min_side=608, max_side=1024):
        image, bboxes, cls, is_crowd, image_id = sample[0], sample[1], sample[2], sample[3], sample[4]

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows%32
        pad_h = 32 - cols%32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        bboxes *= scale

        return torch.from_numpy(new_image), torch.from_numpy(bboxes), torch.from_numpy(cls), is_crowd, image_id, scale

class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):

        sample[0] = ((sample[0].astype(np.float32)-self.mean)/self.std)

        return sample

class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
 
def collater(data):
    imgs = [s[0] for s in data]
    bboxes = [s[1] for s in data]
    cls = [s[2] for s in data]
    is_crowd = [s[3] for s in data]
    image_id = [s[4] for s in data]
    resize_factor = [s[5] for s in data]
        
    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(c.shape[0] for c in bboxes)
    
    if max_num_annots > 0:

        bboxes_padded = torch.ones((len(bboxes), max_num_annots, 4)) * -1
        cls_padded = torch.ones((len(bboxes), max_num_annots, 1)) * -1

        if max_num_annots > 0:
            for idx in range(len(bboxes)):
                bbox = bboxes[idx]
                cl = cls[idx]
                if bbox.shape[0] > 0:
                    bboxes_padded[idx, :bbox.shape[0], :] = bbox
                    cls_padded[idx, :cl.shape[0], :] = cl
    else:
        bboxes_padded = torch.ones((len(bboxes), 1, 5)) * -1
        cls_padded = torch.ones((len(bboxes), 1, 5)) * -1


    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return padded_imgs, cls_padded.to(torch.int64), bboxes_padded, is_crowd, image_id, resize_factor


