# --------------------------------------------------------
# Fine Refine Online Gushing
# Copyright (c) 2018 KAUST IVUL
# Licensed under The MIT License [see LICENSE for details]
# Written by Frost XU
# --------------------------------------------------------

"""The layer used during training to get proposal labels for classifier refinement.

FrogLayer implements a tensorflow Python layer.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from model.config import cfg
from model.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps

DEBUG = False


def frog_layer(boxes, cls_prob, im_labels):
    """Get blobs and copy them into this layer's top blob vector."""

    """Get proposals with highest score."""
    im_labels = im_labels[:, 1:]  # remove bg
    boxes = boxes[:, 1:]
    if DEBUG:
        print('im_labels', im_labels.shape)
    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    im_labels_tmp = im_labels[0, :]
    gt_boxes = np.zeros((0, 4), dtype=np.float32)
    gt_classes = np.zeros((0, 1), dtype=np.int32)
    gt_scores = np.zeros((0, 1), dtype=np.float32)
    for i in xrange(num_classes):
        if im_labels_tmp[i] == 1:
            cls_prob_tmp = cls_prob[:, i].copy()
            max_index = np.argmax(cls_prob_tmp)

            if DEBUG:
                print('max_index:', max_index, 'num_classes:', num_classes)
                print('boxes:', boxes.shape, 'cls_prob_tmp', cls_prob_tmp.shape)

            gt_boxes = np.vstack((gt_boxes, boxes[max_index, :].reshape(1, -1)))
            gt_classes = np.vstack((gt_classes, (i + 1) * np.ones((1, 1), dtype=np.int32)))
            gt_scores = np.vstack((gt_scores,
                                   cls_prob_tmp[max_index] * np.ones((1, 1), dtype=np.float32)))
            cls_prob[max_index, :] = 0

    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)

    overlaps = bbox_overlaps(
        np.ascontiguousarray(boxes, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_classes[gt_assignment, 0]
    cls_loss_weights = gt_scores[gt_assignment, 0]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where(max_overlaps < cfg.TRAIN.FG_THRESH)[0]


    labels[bg_inds] = 0

    if DEBUG:
        print('label', labels.shape, 'weight', cls_loss_weights.shape)
    return labels, cls_loss_weights



#
# def _get_highest_score_proposals(boxes, cls_prob, im_labels):
#     """Get proposals with highest score."""
#
#     num_images, num_classes = im_labels.shape
#     assert num_images == 1, 'batch size shoud be equal to 1'
#     im_labels_tmp = im_labels[0, :]
#     gt_boxes = np.zeros((0, 4), dtype=np.float32)
#     gt_classes = np.zeros((0, 1), dtype=np.int32)
#     gt_scores = np.zeros((0, 1), dtype=np.float32)
#     for i in xrange(num_classes):
#         if im_labels_tmp[i] == 1:
#             cls_prob_tmp = cls_prob[:, i].copy()
#             max_index = np.argmax(cls_prob_tmp)
#
#             if DEBUG:
#                 print( 'max_index:', max_index, 'cls_prob_tmp:', cls_prob_tmp[max_index])
#
#             gt_boxes = np.vstack((gt_boxes, boxes[max_index, :].reshape(1, -1)))
#             gt_classes = np.vstack((gt_classes, (i + 1) * np.ones((1, 1), dtype=np.int32)))
#             gt_scores = np.vstack((gt_scores,
#                                    cls_prob_tmp[max_index] * np.ones((1, 1), dtype=np.float32)))
#             cls_prob[max_index, :] = 0
#
#     proposals = {'gt_boxes' : gt_boxes,
#                  'gt_classes': gt_classes,
#                  'gt_scores': gt_scores}
#
#     return proposals
#
# def _sample_rois(all_rois, proposals):
#     """Generate a random sample of RoIs comprising foreground and background
#     examples.
#     """
#     # overlaps: (rois x gt_boxes)
#     gt_boxes = proposals['gt_boxes']
#     gt_labels = proposals['gt_classes']
#     gt_scores = proposals['gt_scores']
#     overlaps = bbox_overlaps(
#         np.ascontiguousarray(all_rois, dtype=np.float),
#         np.ascontiguousarray(gt_boxes, dtype=np.float))
#     gt_assignment = overlaps.argmax(axis=1)
#     max_overlaps = overlaps.max(axis=1)
#     labels = gt_labels[gt_assignment, 0]
#     cls_loss_weights = gt_scores[gt_assignment, 0]
#
#     # Select foreground RoIs as those with >= FG_THRESH overlap
#     fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
#
#     # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
#     bg_inds = np.where(max_overlaps < cfg.TRAIN.FG_THRESH)[0]
#
#     if DEBUG:
#         print( "number of fg:", len(fg_inds), 'number of bg:', len(bg_inds) )
#
#     labels[bg_inds] = 0
#
#     rois = all_rois
#
#     return labels, rois, cls_loss_weights
