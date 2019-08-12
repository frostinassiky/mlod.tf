# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np

try:
    import cPickle as pickle
except ImportError:
    import pickle
import os
import math

from utils.timer import Timer
from utils.blob import im_list_to_blob

from model.config import cfg, get_output_dir
from model.bbox_transform import clip_boxes, bbox_transform_inv
from model.nms_wrapper import nms

import scipy.io as sio


def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def _get_blobs(im):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {}
    blobs['data'], im_scale_factors = _get_image_blob(im)

    return blobs, im_scale_factors


def _clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes


def _rescale_boxes(boxes, inds, scales):
    """Rescale boxes according to image rescaling."""
    for i in range(boxes.shape[0]):
        boxes[i, :] = boxes[i, :] / scales[int(inds[i])]

    return boxes


def im_detect(sess, net, im):
    blobs, im_scales = _get_blobs(im)
    assert len(im_scales) == 1, "Only single-image batch implemented"

    im_blob = blobs['data']
    blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)

    _, scores, bbox_pred, rois = net.test_image(sess, blobs['data'], blobs['im_info'])

    boxes = rois[:, 1:5] / im_scales[0]
    scores = np.reshape(scores, [scores.shape[0], -1])
    bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = _clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    return scores, pred_boxes


def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    for cls_ind in range(num_classes):
        for im_ind in range(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue

            x1 = dets[:, 0]
            y1 = dets[:, 1]
            x2 = dets[:, 2]
            y2 = dets[:, 3]
            scores = dets[:, 4]
            inds = np.where((x2 > x1) & (y2 > y1))[0]
            dets = dets[inds, :]
            if dets == []:
                continue

            keep = nms(dets, thresh)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes


def test_net_train(sess, net, imdb, weights_filename, max_per_image=100, thresh=0.):
    """Test an FROG network on an image database,
        and generate pseudo ground truths for training faster rcnn."""
    np.random.seed(cfg.RNG_SEED)
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #  all_boxes[cls][image] = N x 5 array of detections in
    #  (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]

    output_dir = get_output_dir(imdb, weights_filename)
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}

    images_real = np.zeros((num_images,), dtype=object)
    gt = np.zeros((num_images,), dtype=object)
    roidb = imdb.roidb

    scores_all = []
    boxes_all = []

    for i in range(num_images):
        im = cv2.imread(imdb.image_path_at(i))

        _t['im_detect'].tic()
        scores, boxes = im_detect(sess, net, im)
        _t['im_detect'].toc()
        scores_all.append(scores)
        boxes_all.append(boxes)

        _t['misc'].tic()

        # skip j = 0, because it's the background class
        for j in range(1, imdb.num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]
            all_boxes[j][i] = cls_dets

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in range(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        gt_tmp = {'aeroplane': np.empty((0, 4), dtype=np.float32),
                  'bicycle': np.empty((0, 4), dtype=np.float32),
                  'bird': np.empty((0, 4), dtype=np.float32),
                  'boat': np.empty((0, 4), dtype=np.float32),
                  'bottle': np.empty((0, 4), dtype=np.float32),
                  'bus': np.empty((0, 4), dtype=np.float32),
                  'car': np.empty((0, 4), dtype=np.float32),
                  'cat': np.empty((0, 4), dtype=np.float32),
                  'chair': np.empty((0, 4), dtype=np.float32),
                  'cow': np.empty((0, 4), dtype=np.float32),
                  'diningtable': np.empty((0, 4), dtype=np.float32),
                  'dog': np.empty((0, 4), dtype=np.float32),
                  'horse': np.empty((0, 4), dtype=np.float32),
                  'motorbike': np.empty((0, 4), dtype=np.float32),
                  'person': np.empty((0, 4), dtype=np.float32),
                  'pottedplant': np.empty((0, 4), dtype=np.float32),
                  'sheep': np.empty((0, 4), dtype=np.float32),
                  'sofa': np.empty((0, 4), dtype=np.float32),
                  'train': np.empty((0, 4), dtype=np.float32),
                  'tvmonitor': np.empty((0, 4), dtype=np.float32)}
        # print(222, roidb[i].keys())
        tmp_idx = np.where(roidb[i]['label'][0][:imdb.num_classes])[0]

        for j in xrange(len(tmp_idx)):
            idx_real = np.argmax(scores[:, tmp_idx[j]])
            gt_tmp[imdb.classes[tmp_idx[j]]] = np.array([boxes[idx_real, tmp_idx[j]*4+1],
                                                         boxes[idx_real, tmp_idx[j]*4],
                                                         boxes[idx_real, tmp_idx[j]*4+3],
                                                         boxes[idx_real, tmp_idx[j]*4+2]], dtype=np.float32)
            gt_tmp[imdb.classes[tmp_idx[j]]] += 1

        gt[i] = {'gt' : gt_tmp}

        images_real[i] = imdb.image_index[i]


        _t['misc'].toc()

        print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time))

    # save gt
    model_save_gt = {'images' : images_real, 'gt' : gt}
    sio.savemat('{}_pseudo_gt.mat'.format(imdb.name), model_save_gt)

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)

