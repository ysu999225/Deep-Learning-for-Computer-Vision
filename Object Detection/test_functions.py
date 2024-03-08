import unittest

import numpy as np
from gradescope_utils.autograder_utils.decorators import weight, visibility, number
from network import RetinaNet
import utils
from detection_utils import compute_targets, compute_bbox_targets, apply_bbox_deltas, nms, compute_bbox_iou
import torch

test_compute_targets_file = './test_data/compute_targets.pkl'
test_compute_bbox_targets_file = './test_data/compute_bbox_targets.pkl'
test_nms_file = './test_data/nms.pkl'
test_apply_bbox_deltas_file = './test_data/apply_bbox_deltas.pkl'
test_generate_anchors_file = './test_data/generate_anchors.pkl'

class TestClass(unittest.TestCase):
    def setUp(self) -> None:
        pass

    #####################
    # Compute Anchors   #
    #####################

    def _test_generate_anchors(self, RetinaNet, test_data_file, rtol=1e-5, atol=1e-8):
        dt = utils.load_variables(test_data_file)
        model = RetinaNet(fpn=True, p67=True)

        for i in range(len(dt)):
            inputs = dt[i]['inputs']
            outputs = dt[i]['outputs']
            image, bbox = inputs['image'], inputs['bbox']
            
            outs = model(image)
            anchors = []
            for _, _, anchor in outs:
                B, A, H, W = anchor.shape
                anchor = anchor.reshape(B, A // 4, -1, H, W)
                anchors.append(anchor.permute(0, 1, 3, 4, 2).reshape(B, -1, 4))
            anchors = torch.cat(anchors, dim=1).squeeze(0)

            iou = compute_bbox_iou(anchors, bbox[0])
            max_iou = max(iou)

            self.assertTrue(
                np.allclose(max_iou, outputs['max_iou'], rtol=rtol, atol=atol),
                f'Output max iou value does not match expected max iou value!'
            )

    @weight(1.0)
    @number("1.1")
    @visibility('visible')
    def test_generate_anchors(self):
        self._test_generate_anchors(RetinaNet, test_generate_anchors_file)


    ######################################################
    # Test  Assignment of GroundTruth Targets to Anchors #
    ######################################################

    def _test_compute_targets(self, compute_targets, test_data_file, rtol=1e-5, atol=1e-8):
        dt = utils.load_variables(test_data_file)
        for i in range(len(dt)):
            inputs = dt[i]['inputs']
            outputs = dt[i]['outputs']
            
            anchor, cls, bbox, is_crowd = inputs['anchor'], inputs['cls'], inputs['bbox'], inputs['is_crowd']
            gt_clss, gt_bboxes = compute_targets(anchor, cls, bbox, is_crowd)


            self.assertTrue(
                gt_clss.shape == outputs['gt_clss'].shape,
                f"Shape of groundtruth classes {gt_clss.shape} does not match expected shape {outputs['gt_clss'].shape}!"
            )

            self.assertTrue(
                np.allclose(gt_clss, outputs['gt_clss'], rtol=rtol, atol=atol),
                f'Output gt_clss does not match expected output for groundtruth classes!'
            )

            self.assertTrue(
                gt_bboxes.shape == outputs['gt_bboxes'].shape,
                f"Shape of groundtruth bounding boxes {gt_bboxes.shape} does not match expected shape {outputs['gt_bboxes'].shape}!"
            )
            self.assertTrue(
                np.allclose(gt_bboxes, outputs['gt_bboxes'], rtol=rtol, atol=atol),
                f'Output gt_bboxes values does not match expected output for groundtruth bounding boxes!'
            )


    @weight(2.0)
    @number("2")
    @visibility('visible')
    def test_compute_targets(self):
        self._test_compute_targets(compute_targets, test_compute_targets_file)


    ########################################################
    # Relative Offset between Anchor and Groundtruth Box   #
    ########################################################

    def _test_compute_bbox_targets(self, compute_bbox_targets, test_data_file, rtol=1e-5, atol=1e-5):
        dt = utils.load_variables(test_data_file)
        for i in range(len(dt)):
            inputs = dt[i]['inputs']
            outputs = dt[i]['outputs']
            anchors, gt_bboxes = inputs['anchors'], inputs['gt_bboxes']
            bbox_reg_target = compute_bbox_targets(anchors, gt_bboxes)

            self.assertTrue(
                bbox_reg_target.shape == outputs['bbox_reg_target'].shape,
                f"Shape of regression offsets {bbox_reg_target.shape} does not match expected shape {outputs['bbox_reg_target'].shape}!"
            )
            self.assertTrue(
                np.allclose(bbox_reg_target[:,0], outputs['bbox_reg_target'][:,0], rtol=rtol, atol=atol),
                f'Output regression offsets values for delta_x do not match expected output regression offsets!'
            )
            self.assertTrue(
                np.allclose(bbox_reg_target[:,1], outputs['bbox_reg_target'][:,1], rtol=rtol, atol=atol),
                f'Output regression offsets values for delta_y do not match expected output regression offsets!'
            )
            self.assertTrue(
                np.allclose(bbox_reg_target[:,2], outputs['bbox_reg_target'][:,2], rtol=rtol, atol=atol),
                f'Output regression offsets values for delta_w do not match expected output regression offsets!'
            )
            self.assertTrue(
                np.allclose(bbox_reg_target[:,3], outputs['bbox_reg_target'][:,3], rtol=rtol, atol=atol),
                f'Output regression offsets values for delta_h do not match expected output regression offsets!'
            )



    @weight(2.0)
    @number("3")
    @visibility('visible')
    def test_compute_bbox_targets(self):
        self._test_compute_bbox_targets(compute_bbox_targets, test_compute_bbox_targets_file)


    ##############################################
    # Obtain New Boxes by Applying Bbox Deltas   #
    ##############################################

    def _test_apply_bbox_deltas(self, apply_bbox_deltas, test_data_file, rtol=1e-4, atol=1e-3):
        dt = utils.load_variables(test_data_file)
        for i in range(len(dt)):
            inputs = dt[i]['inputs']
            outputs = dt[i]['outputs']
            boxes, deltas = inputs['boxes'], inputs['deltas']
            new_boxes = apply_bbox_deltas(boxes, deltas)

            self.assertTrue(
                new_boxes.shape == outputs['new_boxes'].shape,
                f"Shape of new boxes {new_boxes.shape} does not match expected shape {outputs['new_boxes'].shape}!"
            )
            self.assertTrue(
                np.allclose(new_boxes, outputs['new_boxes'], rtol=rtol, atol=atol),
                f'Output new boxes values does not match expected output new boxes!'
            )


    @weight(2.0)
    @number("4")
    @visibility('visible')
    def test_apply_bbox_deltas(self):
        self._test_apply_bbox_deltas(apply_bbox_deltas, test_apply_bbox_deltas_file)


    ####################################
    # Test Non-Maximum Suppression     #
    ####################################

    def _test_nms(self, nms, test_data_file, rtol=1e-5, atol=1e-8):
        dt = utils.load_variables(test_data_file)
        for i in range(len(dt)):
            inputs = dt[i]['inputs']
            outputs = dt[i]['outputs']
            bboxes, scores, threshold = inputs['bboxes'], inputs['scores'], inputs['threshold']
            keep = nms(bboxes, scores, threshold)

            self.assertTrue(
                keep.shape == outputs['keep'].shape,
                f"Shape of keep {keep.shape} does not match expected shape {outputs['keep'].shape}!"
            )
            self.assertTrue(
                np.allclose(keep, outputs['keep'], rtol=rtol, atol=atol),
                f'Output keep does not match expected output!'
            )

    @weight(2.0)
    @number("5")
    @visibility('visible')
    def test_nms(self):
        self._test_nms(nms, test_nms_file)

if __name__ == '__main__':
    unittest.main()
