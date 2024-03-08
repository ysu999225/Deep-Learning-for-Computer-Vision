import torch
import numpy as np
import random

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    
def get_detections(outs):
    B, BB, _, _ = outs[0][0].shape
    _, A, _, _ = outs[0][2].shape
    A = A // 4
    num_classes = BB // A
    
    pred_bboxes, pred_clss, anchors = [], [], []
    for pred_cls, pred_bbox, anchor in outs:
        # Get all the anchors, pred and bboxes
        H, W = pred_cls.shape[-2:]
        pred_cls = pred_cls.reshape(B, A, -1, H, W)
        pred_bbox = pred_bbox.reshape(B, A, -1, H, W)

        pred_clss.append(pred_cls.permute(0, 1, 3, 4, 2).reshape(B, -1, num_classes))
        pred_bboxes.append(pred_bbox.permute(0, 1, 3, 4, 2).reshape(B, -1, 4))

        anchor = anchor.reshape(B, A, -1, H, W)
        anchors.append(anchor.permute(0, 1, 3, 4, 2).reshape(B, -1, 4))
    pred_clss = torch.cat(pred_clss, dim=1)
    pred_bboxes = torch.cat(pred_bboxes, dim=1)
    anchors = torch.cat(anchors, dim=1)
    return pred_clss, pred_bboxes, anchors

def compute_bbox_iou(bbox1, bbox2, dim=1):
    """
    Args:
        bbox1: (N, 4) tensor of (x1, y1, x2, y2)
        bbox2: (M, 4) tensor of (x1, y1, x2, y2)
    Returns
        iou: (N, M) tensor of IoU values
    """
    bbox1 = bbox1.unsqueeze(1)
    bbox2 = bbox2.unsqueeze(0)
    max_min_x = torch.max(bbox1[...,0], bbox2[...,0])
    min_max_x = torch.min(bbox1[...,2], bbox2[...,2])
    max_min_y = torch.max(bbox1[...,1], bbox2[...,1])
    min_max_y = torch.min(bbox1[...,3], bbox2[...,3])
    intersection = torch.clamp(min_max_x - max_min_x, min=0) * torch.clamp(min_max_y - max_min_y, min=0)
    area1 = (bbox1[...,2] - bbox1[...,0]) * (bbox1[...,3] - bbox1[...,1])
    area2 = (bbox2[...,2] - bbox2[...,0]) * (bbox2[...,3] - bbox2[...,1])
    iou = intersection / (area1 + area2 - intersection)
    return iou

def compute_targets(anchor, cls, bbox, is_crowd):
    """
    Args:
        anchors: batch of anchors in the format (x1, y1, x2, y2) or in other words (xmin, ymin, xmax, ymax); shape is (B, A, 4), where B denotes image batch size and A denotes the number of anchors
        cls: groundtruth object classes of shape (B, number of objects in the image, 1)
        bbox: groundtruth bounding boxes of shape (B, number of objects in the image, 4)
    Returns:
        gt_clss: groundtruth class targets of shape (B, A, 1)
        gt_bboxes: groundtruth bbox targets of shape (B, A, 4)
    
    Hint: remember if the max_iou for that bounding box is between [0 0.4) then the gt_cls should equal 0(because it is being assigned as background) and the
    gt_bbox should be all zero(it can be anything since it will be ignored however our tests set them to zero so you should too).
    Also, if the max iou is between [0.4, 0.5) then the gt_cls should be equal to -1(since it's neither background or assigned to a class. This is basically tells the model to ignore this box) 
    and the gt_bbox should again arbitrarilarly be set to all zeros).
    Otherwise if the max_iou > 0.5, you should assign the anchor to the gt_box with the max iou, and the gt_cls will be the ground truth class of that max_iou box
    Hint: use torch.max to get both the max iou and the index of the max iou.

    Hint: We recommend using the compute_bbox_iou function which efficently computes the ious between two lists of bounding boxes as a helper function.

    Hint: make sure that the returned gt_clss tensor is of type int(since it will be used as an index in the loss function). Also make sure that both the gt_bboxes and gt_clss are on the same device as the anchor. 
    You can do this by calling .to(anchor.device) on the tensor you want to move to the same device as the anchor.

    VECTORIZING CODE: Again, you can use for loops initially to make the tests pass, but In order to make your code efficient 
    during training, you should only have one for loop over the batch dimension and everything else should be vectorized. We recommend using boolean masks to do this. i.e
    you can compute the max_ious for all the anchor boxes and then do gt_cls[max_iou < 0.4] = 0 to access all the anchor boxes that should be set to background and setting their gt_cls to 0. 
    This will remove the need for a for loop over all the anchor boxes. You can then do the same for the other cases. This will make your code much more efficient and faster to train.
    """
   # TODO(student): Complete this function
    #Extract the batch size B, the number of anchors and the number of ground objects in image
    B, A = anchor.shape[:2]
    O = bbox.size(1)
    device = anchor.device
    # Initialize both classes and bboxes to 0
    #Index put requires the source and destination dtypes match, got Float for the destination and Int for the source.
    gt_clss = torch.zeros(B, A, 1, dtype=torch.int).to(device)
    gt_bboxes = torch.zeros(B, A, 4).to(device)
    # Compute IoUs
    #(B,A,4) to (B,A,1,4)
    reshaped_anchors = anchor.unsqueeze(2).to(device)
    #(B,0,4) to (B,1,0,4)
    reshaped_bbox = bbox.unsqueeze(1).to(device)
    #calculate the IoU between each pair and reshape back
    iou = compute_bbox_iou(reshaped_anchors, reshaped_bbox).view(B,A,O)
    for b in range(B):
        # For each anchor, find maximum IoU
        max, idx = iou[b].max(dim=1)
        # Assign class 0 to max_iou < 0.4
        gt_clss[b][max < 0.4] = 0
        # Assign class -1 to [0.4,0.5)
        gt_clss[b][(max >= 0.4) & (max < 0.5)] = -1
        # Assign ground truth class and bbox to max_iou >= 0.5
        high = max >= 0.5
        #assigning the ground truth class to anchors
        # value tensor of shape [105, 1] cannot be broadcast to indexing result of shape [105]
        # add 0 index to remove the additional dimension        
        gt_clss[b, high, 0] =  cls[b, idx[high], 0].to(torch.int).to(device)
        #assigning the ground truth bbox class to anchors
        gt_bboxes[b, high] = bbox[b, idx[high]].to(device)
    return gt_clss.to(torch.int), gt_bboxes

def compute_bbox_targets(anchors, gt_bboxes):
    """
    Args:
        anchors: anchors of shape (A, 4)
        gt_bboxes: groundtruth object classes of shape (A, 4)
    Returns:
        bbox_reg_target: regression offset of shape (A, 4)
    
    Remember that the delta_x and delta_y we are computing is in regards to the center of the anchor box. I.E, we're seeing how much that center of the anchor box changes. 
    We also need to normalize delta_x and delta_y which means that we need to divide them by the width or height of the anchor box respectively. This is to make
    our regression targets more invariant to the size of the original anchor box. So, this means that:
    delta_x = (gt_bbox_center_x - anchor_center_x) / anchor_width  and delta_y would be computed in a similar manner.

    When computing delta_w and delta_h, there are a few things to note.
    1. We also want to normalize these with respect to the width and height of the anchor boxes. so delta_x = gt_bbox_width / anchor_width
    2. Logarithm: In order to make our regresssion targets better handle varying sizees of the bounding boxes, we use the logarithmic scale for our delta_w and delta_h
       This is to ensure that if for example the gt_width is twice or 1/2 the size of the anchor_width, the magnitude in the log scale would stay the same but only the sign of
       our regression target would be different. Therefore our formula changes to delta_x = log(gt_bbox_width / anchor_width)
    3. Clamping: Remember that logarithms can't handle negative values and that the log of values very close to zero will have very large magnitudes and have extremly 
       high gradients which might make training unstable. To mitigate this we use clamping to ensure that the value that we log isn't too small. Therefore, our final formula will be
       delta_w = log(max(gt_bbox_width,1) / anchor_width)
       
    """
    # TODO(student): Complete this function
    # get the anchor centers and widths and heights
    #[x1,y1,x2,y2]
    w =  anchors[:, 2] - anchors[:, 0]
    h =  anchors[:, 3] - anchors[:, 1]
    centerx = (anchors[:, 0] + anchors[:, 2]) / 2
    centery = (anchors[:, 1] + anchors[:, 3]) / 2
    
    #get the ground truth bounding box centers and w and h
    gt_w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
    gt_h= gt_bboxes[:, 3] - gt_bboxes[:, 1]
    gt_centerx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2
    gt_centery = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2
    
    # get the delta_x, delta_y, delta_w, delta_h
    #RuntimeError: torch.clamp: At least one of 'min' or 'max' must not be None
    delta_x = (gt_centerx - centerx) / w
    delta_y = (gt_centery - centery) / h
    # torch.maximum(a, torch.tensor(1)) 
    # max(gt_bbox_width,1)
    result_w = torch.maximum(gt_w, torch.tensor(1.))
    delta_w = torch.log(result_w/w)
    #same as above
    result_h = torch.maximum(gt_h, torch.tensor(1.))
    delta_h = torch.log(result_h / h)
    return torch.stack([delta_x, delta_y, delta_w, delta_h],dim = -1)

def apply_bbox_deltas(boxes, deltas):
    """
    Args:
        boxes: (N, 4) tensor of (x1, y1, x2, y2)
        deltas: (N, 4) tensor of (dxc, dyc, dlogw, dlogh)
    Returns
        boxes: (N, 4) tensor of (x1, y1, x2, y2)
        
    """
    # TODO(student): Complete this function

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    w = boxes[:,2] - boxes[:,0]
    h = boxes[:,3] - boxes[:,1]
    
    cx = boxes[:, 0] + 0.5 * w
    cy = boxes[:, 1] + 0.5 * h
    
    cx += deltas[:, 0] * w
    cy += deltas[:, 1] * h
    w *= torch.exp(deltas[:, 2])
    h *= torch.exp(deltas[:, 3])
    
    #boxes: (N, 4) tensor of (x1, y1, x2, y2)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    
    new_boxes = torch.stack([x1, y1, x2, y2], axis=1)
    return new_boxes

def nms(bboxes, scores, threshold=0.5):
    """
    Args:
        bboxes: (N, 4) tensor of (x1, y1, x2, y2)
        scores: (N,) tensor of scores
    Returns:
        keep: (K,) tensor of indices to keep
    
    Remember that nms is there to prevent having many boxes that overlap eachother. To do this, if multiple boxes overlap eachother beyond a
    threshold iou, nms will pick the "best" box(the one with the highest score) and remove the rest. One way to implement this is to
    first compute the ious between all pairs of bboxes. Then loop over the bboxes from highest score to lowest score. Since this is the 
    best bbox(the one with the highest score), It will be choosen over all overlapping boxes. Therefore, you should add this bbox to your final 
    resulting bboxes and remove all the boxes that overlap with it from consideration. Then repeat until you've gone through all of the bboxes.

    make sure that the indices tensor that you return is of type int or long(since it will be used as an index to select the relevant bboxes to output)
    """
    # TODO(student): Complete this function
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    
    # area
    areas = (x2 - x1) * (y2 - y1)
    # descending order to help find the best
    order = torch.argsort(scores, descending=True)
    # initialize the list of keep
    keep = []
    #  first compute the ious between all pairs of bboxes. 
    iou = compute_bbox_iou(bboxes, bboxes)
    #Then loop over the bboxes from highest score to lowest score
    while order.numel() > 0:
        #0 index is the highest
        i = order[0]
        keep.append(i)
        # reach the lowest then break
        # the last one-- only one elements
        if order.size(0) == 1:
            break
        # highest score box
        high = bboxes[i]
        #(N-1,4)
        ious = compute_bbox_iou(high.unsqueeze(0), bboxes[order[1:]]).squeeze(0)
        #compare with the 0.5
        indices = torch.nonzero(ious < 0.5).squeeze(1)
        #you should add this bbox to your final 
        order = order[indices + 1]
        #Shape of keep torch.Size([1]) does not match expected shape torch.Size([31])
    #return torch.tensor(keep)
    return torch.tensor(keep, dtype=torch.long)
        
    
