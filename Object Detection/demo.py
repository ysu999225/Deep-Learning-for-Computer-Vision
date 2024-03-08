import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from network import RetinaNet
from detection_utils import compute_targets, get_detections, set_seed
from predict import validate, test
from tensorboardX import SummaryWriter
from absl import app, flags
from tqdm import tqdm
import numpy as np
from dataset import CocoDataset, Resizer, Normalizer, collater
from torchvision import transforms
import losses

FLAGS = flags.FLAGS
flags.DEFINE_float('lr', 1e-4, 'Learning Rate')
flags.DEFINE_float('momentum', 0.9, 'Momentum for optimizer')
flags.DEFINE_float('weight_decay', 1e-4, 'Weight Deacy for optimizer')
flags.DEFINE_string('output_dir', 'runs/retina-net-basic/', 'Output Directory')
flags.DEFINE_integer('batch_size', 1, 'Batch Size')
flags.DEFINE_integer('seed', 2, 'Random seed')
flags.DEFINE_integer('max_iter', 120000, 'Total Iterations')
flags.DEFINE_multi_integer('lr_step', [60000, 100000], 'Iterations to reduce learning rate')

def main(_):
    torch.manual_seed(FLAGS.seed)
    set_seed(FLAGS.seed)
    dataset_train = CocoDataset('train', seed=FLAGS.seed,
        transform=transforms.Compose([Normalizer(), Resizer()]))
    dataset_val = CocoDataset('val', seed=0, 
        transform=transforms.Compose([Normalizer(), Resizer()]))
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater) 
    
    model = RetinaNet(p67=True, fpn=True)

    num_classes = dataset_train.num_classes
    device = torch.device('cuda:0')
    model.to(device)


    writer = SummaryWriter(FLAGS.output_dir, flush_secs=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=FLAGS.lr, 
                                momentum=FLAGS.momentum, 
                                weight_decay=FLAGS.weight_decay)
    
    milestones = [int(x) for x in FLAGS.lr_step]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1)
    
    optimizer.zero_grad()
    dataloader_iter = None
    pbar = tqdm(range(FLAGS.max_iter))
    
    cls_loss_np, bbox_loss_np, total_loss_np = [], [], []

    lossFunc = losses.LossFunc()
    
    for i in pbar:
        if dataloader_iter is None or i % len(dataloader_iter) == 0:
            dataloader_iter = iter(dataloader_train)
        
        image, cls, bbox, is_crowd, image_id, _ = next(dataloader_iter)

        if len(bbox) == 0:
            continue

        image = image.to(device)
        bbox = bbox.to(device)
        cls = cls.to(device)

        outs = model(image)
        pred_clss, pred_bboxes, anchors = get_detections(outs)
        gt_clss, gt_bboxes = compute_targets(anchors, cls, bbox, is_crowd)
        
        pred_clss = pred_clss.sigmoid()
        classification_loss, regression_loss = lossFunc(pred_clss, pred_bboxes, anchors, gt_clss, gt_bboxes)
        cls_loss = classification_loss.mean()
        bbox_loss = regression_loss.mean()
        total_loss = cls_loss + bbox_loss
        
        if np.isnan(total_loss.item()):
            print('Loss went to NaN')
            break
        
        if np.isinf(total_loss.item()):
            print('Loss went to Inf')
            break
        
        total_loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        # Some logging
        lr = scheduler.get_last_lr()[0]
        total_loss_np.append(total_loss.item())
        cls_loss_np.append(cls_loss.item())
        bbox_loss_np.append(bbox_loss.item())
                      
        if (i+1) % 20 == 0:
            writer.add_scalar('loss_box_reg', np.mean(bbox_loss_np), i+1)
            writer.add_scalar('lr', lr, i+1)
            writer.add_scalar('loss_cls', np.mean(cls_loss_np), i+1)
            writer.add_scalar('total_loss', np.mean(total_loss_np), i+1)
            pbar.set_description(f"{i+1} / {lr:5.6f} / {np.mean(cls_loss_np):5.3f} / {np.mean(bbox_loss_np):5.3f} / {np.mean(total_loss_np):5.3f}")
            cls_loss_np, bbox_loss_np, total_loss_np = [], [], []


        if (i+1) % 100000 == 0:
            torch.save(model.state_dict(), f'{FLAGS.output_dir}/model_{i+1}.pth')
            
        if (i+1) % 5000 == 0 or (i+1) == len(pbar):
            print('Validating...')
            val_dataloader = DataLoader(dataset_val, num_workers=3, collate_fn=collater)
            result_file_name = f'{FLAGS.output_dir}/results_{i+1}_val.json'
            model.eval()
            validate(dataset_val, val_dataloader, device, model, result_file_name, writer, i+1)
            model.train()

    torch.save(model.state_dict(), f'{FLAGS.output_dir}/model_final.pth')

    # Save prediction result on test set
    dataset_test = CocoDataset('test', transform=transforms.Compose([Normalizer(), Resizer()]))
    test_dataloader = DataLoader(dataset_test, num_workers=1, collate_fn=collater)
    result_file_name = f'{FLAGS.output_dir}/results_{FLAGS.max_iter}_test.json'
    model.eval()
    test(dataset_test, test_dataloader, device, model, result_file_name)

if __name__ == '__main__':
    app.run(main)
