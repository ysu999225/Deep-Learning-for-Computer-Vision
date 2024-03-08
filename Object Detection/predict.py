import os
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=6

import torch
from torch import nn
import torchvision
from dataset import CocoDataset
from torch.utils.data import DataLoader
from network import RetinaNet
from detection_utils import compute_targets, compute_bbox_targets, get_detections, apply_bbox_deltas, nms
from tensorboardX import SummaryWriter
from absl import app, flags
from tqdm import tqdm
import json
import numpy as np
from dataset import CocoDataset, collater, Resizer, Normalizer
from torchvision import transforms

FLAGS = flags.FLAGS
flags.DEFINE_float('test_nms_thresh', 0.5, 'Threshold for NMS')
flags.DEFINE_string('test_set', 'test', 'What set to test on')
flags.DEFINE_float('test_score_thresh', 0.05, 'Minimum score to decode detections')
flags.DEFINE_integer('test_det_thresh', 100, 'Maximum detections to decode')
flags.DEFINE_integer('test_model_checkpoint', 85000, 'Which snapshot to load')
flags.DEFINE_string('model_dir', 'runs/retina-net-basic', 'Output Directory')

def predict_image(image, image_id, resize_factor, model, cat_map):
    model.eval()
    results = []

    with torch.no_grad():
        outs = model(image)

        pred_clss, pred_bboxes, anchors = get_detections(outs)
        prob_clss = pred_clss.sigmoid()
        prob_clss = prob_clss[0,...]
        pred_bboxes = pred_bboxes[0,...]
        anchors = anchors[0,...]
        out_bboxes = apply_bbox_deltas(anchors, pred_bboxes)
        out_bboxes = out_bboxes / resize_factor[0]
        
        for j in range(prob_clss.shape[1]):
            prob_cls = prob_clss[:,j]
            out_bbox = out_bboxes + 0.
            anchor = anchors + 0.
            keep = prob_cls > FLAGS.test_score_thresh
            prob_cls = prob_cls[keep]
            out_bbox = out_bbox[keep,:]
            anchor = anchor[keep,:]
            keep = nms(anchor, prob_cls, FLAGS.test_nms_thresh)
            prob_cls = prob_cls[keep]
            out_bbox = out_bbox[keep,:]
            anchor = anchor[keep,:]
            keep = torch.argsort(prob_cls, descending=True)[:FLAGS.test_det_thresh]
            prob_cls = prob_cls[keep]
            out_bbox = out_bbox[keep,:]
            anchor = anchor[keep,:]
            
            for a, p in zip(out_bbox.cpu().double().numpy(), 
                            prob_cls.cpu().double().numpy()):
                res = {}
                res['image_id'] = int(image_id[0])
                res['category_id'] = cat_map.index(j+1)
                res['bbox'] = [a[0], a[1], a[2]-a[0], a[3]-a[1]]
                res['score'] = p
                results.append(res)
    return results

def validate(dataset, dataloader, device, model, 
             result_file_name, writer, iteration):
    model.eval()

    results = []
    for i, (image, _, _, _, image_id, resize_factor) in enumerate(tqdm(dataloader)):
        image = image.to(device)
        results += predict_image(image, image_id, resize_factor, model, 
                                 list(dataset.cat_map))
    if len(results) > 0:
        json.dump(results, open(result_file_name, 'w'))
        metrics, classes = dataset.evaluate(result_file_name)
        print_results(metrics, classes)
        log_results(writer, metrics, classes, iteration)
    else:
        print(f'No detections above detection threshold of {FLAGS.test_score_thresh}, skipping evaluation.')

def test(dataset, dataloader, device, model, result_file_name):
    model.eval()
    
    results = []
    for i, (image, _, _, _, image_id, resize_factor) in enumerate(tqdm(dataloader)):
        image = image.to(device)
        results += predict_image(image, image_id, resize_factor, model, 
                                 list(dataset.cat_map))
    
    json.dump(results, open(result_file_name, 'w'))


def main(_):
    dataset = CocoDataset(FLAGS.test_set, transform=transforms.Compose([Normalizer(), Resizer()]))
    sampler = None
    dataloader = DataLoader(dataset, num_workers=1, collate_fn=collater, batch_sampler=sampler)

    model = RetinaNet(fpn=True, p67=True)
    model.load_state_dict(
        torch.load(
            f'{FLAGS.model_dir}/model_{FLAGS.test_model_checkpoint}.pth'))
    model.eval()

    device = torch.device('cuda:0')
    model.to(device)
    results = []
    for i, (image, _, _, _, image_id, resize_factor) in enumerate(tqdm(dataloader)):
        image = image.to(device)
        results += predict_image(image, image_id, resize_factor, model, list(dataset.cat_map))
    
    results_file_name = f'{FLAGS.model_dir}/results_{FLAGS.test_model_checkpoint}_{FLAGS.test_set}.json'
    json.dump(results, open(results_file_name, 'w'))
    # metrics, classes = dataset.evaluate(results_file_name)
    # print_results(metrics, classes)

def print_results(metrics, classes):
        tt = ''
        print(f'{tt:10s}\tAP\tAP50\tAP75\tAPs\tAPm\tAPl')
        for i, (m, c) in enumerate(zip(metrics, classes)):
            print(f'{c:>10s}\t{m[0]:.3f}\t{m[1]:.3f}\t{m[2]:.3f}\t{m[3]:.3f}\t{m[4]:.3f}\t{m[5]:.3f}')

def log_results(writer, metrics, classes, iteration):
     for i, (m, c) in enumerate(zip(metrics, classes)):
        if i == 0:
            writer.add_scalar(f'AP', m[0], iteration)
            writer.add_scalar(f'AP50', m[1], iteration)
            writer.add_scalar(f'AP75', m[2], iteration)
            for j, tag in enumerate(['', '50', '75', 's', 'm', 'l']):
                writer.add_scalar(f'bbox/AP{tag}', m[j]*100, iteration)
        else:
            writer.add_scalar(f'AP/{c}', m[0], iteration)
            writer.add_scalar(f'AP50/{c}', m[1], iteration)
            writer.add_scalar(f'AP75/{c}', m[2], iteration)
            writer.add_scalar(f'bbox/AP-{c}', m[0]*100, iteration)

if __name__ == '__main__':
    app.run(main)
