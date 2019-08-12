# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# Modified by Frost
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import pickle
import subprocess
import uuid
from .voc_eval import voc_eval
from model.config import cfg
from scipy.sparse import vstack
from utils.cython_bbox import bbox_overlaps

class pascal_voc(imdb):
    def __init__(self, image_set, year, use_diff=False):
        name = 'voc_' + year + '_' + image_set
        if use_diff:
            name += '_diff'
        imdb.__init__(self, name)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path()
        self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
        self._classes = ('__background__',  # always index 0
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
        self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': use_diff,
                       'matlab_eval': False,
                       'rpn_file': None}

        assert os.path.exists(self._devkit_path), \
            'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

        # image-level labels
        self._image_label_txt = []


    def image_path_at(self, i):
        """
    Return the absolute path to image i in the image sequence.
    """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
    Construct an image path from the image's "index" identifier.
    """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
    Load the indexes listed in this dataset's image set file.
    """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
    Return the default path where PASCAL VOC is expected to be installed.
    """
        return os.path.join(cfg.DATA_DIR, 'VOCdevkit' + self._year)

    def gt_roidb(self):
        """
    Return the database of ground-truth regions of interest.

    This function loads/saves from/to a cache file to speed up future calls.
    """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                try:
                    roidb = pickle.load(fid)
                except:
                    roidb = pickle.load(fid, encoding='bytes')
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        if self._image_set != 'test':
            gt_roidb_mat = self._load_annotation(self.image_index)
            gt_roidb_xml = [self._load_pascal_annotation(index)
                        for index in self.image_index]


            gt_roidb = []

            def combine_gts(gt1, gt2):
                # gt1 is pseudo box
                # boxes = np.concatenate((gt1['boxes'], gt2['boxes']), axis=0)
                #boxes_vis = np.concatenate((gt1['boxes_vis'], gt2['boxes_vis']), axis=0)
                gt_boxes = gt2['boxes']
                boxes = gt1['boxes']
                if len(gt_boxes)==0:
                    keep_inds = range(len(boxes))
                else:
                    inner_overlaps = bbox_overlaps(
                        np.ascontiguousarray(boxes, dtype=np.float),
                        np.ascontiguousarray(gt_boxes, dtype=np.float))
                    # gt_assignment = overlaps.argmax(axis=1)
                    max_overlaps = inner_overlaps.max(axis=1)
                    keep_inds = np.where(max_overlaps < cfg.TRAIN.P_FG_THRESH)[0] # keep boxes which have small / no overlap
                DEBUG = True
                if DEBUG:
                    if len(keep_inds) != len(boxes):
                        print('From '+str(len(boxes)) + ' choose ' + str(keep_inds))

                boxes = np.concatenate((gt1['boxes'][keep_inds], gt2['boxes']), axis=0)
                gt_classes = np.concatenate((gt1['gt_classes'][keep_inds], gt2['gt_classes']), axis=0)

                is_pseudo = np.concatenate((np.ones(gt1['gt_classes'][keep_inds].shape), np.zeros(gt2['gt_classes'].shape)), axis=0)
                # not_pseudo = np.concatenate((np.zeros(gt1['boxes'].shape), np.ones(gt2['boxes'].shape)), axis=0)

                overlaps = vstack([gt1['gt_overlaps'][keep_inds], gt2['gt_overlaps']]).todense()
                overlaps = scipy.sparse.csr_matrix(overlaps)
                return dict(boxes=boxes,
                            gt_classes=gt_classes,
                            gt_overlaps=overlaps,
                            flipped=False,
                            pseudo=is_pseudo,
                            label=gt2['label'])


            for i in xrange(len(gt_roidb_mat)):
                if True:  # combine two gts
                    roi = combine_gts(gt_roidb_mat[i], gt_roidb_xml[i])
                    gt_roidb.append(combine_gts(gt_roidb_mat[i], gt_roidb_xml[i]))
                else:  # only use missed gt
                    gt_roidb.append(gt_roidb_xml[i])
        else:
            gt_roidb = [self._load_pascal_annotation(index)
                        for index in self.image_index]

        '''

    gt_roidb = [self._load_pascal_annotation(index)
                for index in self.image_index]
    '''
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def _load_annotation(self, indexs):
        """
        Load imge gt from mat file
        """
        image_gt_file = os.path.join(self.cache_path, '..', 'pseudo',
                                     'voc_' + self._year + '_' + self._image_set + '_gt.mat')
        assert os.path.isfile(image_gt_file), 'Error no gt_mat file...' + image_gt_file
        raw_data = sio.loadmat(image_gt_file)
        image_list = raw_data['images']
        image_gt = raw_data['gt']
        assert (image_list.shape[1] == len(indexs)), 'gt num not equal to imges list'

        gt_roidb = []
        for idx, index in enumerate(indexs):
            print(idx)
            print(image_list[0, idx])
            print(index)
            assert image_list[0, idx] == index, 'the gt order is not same with txt file'

            boxes = np.zeros((0, 4), dtype=np.uint16)
            gt_classes = np.zeros((0), dtype=np.int32)
            overlaps = np.zeros((0, self.num_classes), dtype=np.float32)

            for cls in self._classes[1:]:
                gt_matrix = image_gt[0, idx]['gt'][0, 0][cls][0, 0]
                gt_shape = gt_matrix.shape
                if gt_shape[1] == 0:
                    continue
                gt_class = np.zeros((gt_shape[0]), dtype=np.int32)
                gt_class[:] = self._class_to_ind[cls]
                overlap = np.zeros((gt_shape[0], self.num_classes), dtype=np.float32)
                overlap[:, self._class_to_ind[cls]] = 1.0
                # gt box in mat is ymin, xmin, ymax, xmax
                # need convert to xmin, ymin, xmax, ymax
                boxes = np.vstack((boxes, gt_matrix[:, [1, 0, 3, 2]] - 1))
                gt_classes = np.hstack((gt_classes, gt_class))
                overlaps = np.vstack((overlaps, overlap))

            overlaps = scipy.sparse.csr_matrix(overlaps)
            gt_roidb.append({'boxes': boxes,
                             #'boxes_vis': boxes,
                             'gt_classes': gt_classes,
                             'gt_overlaps': overlaps,
                             'flipped': False})
        return gt_roidb

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print('loading {}'.format(filename))
        assert os.path.exists(filename), \
            'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = pickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_pascal_labels(self, index):
        """
        Load label from 20 txt file
        :param index: id of image
        :return: 20+1 array
        """
        if len(self._image_label_txt)==0:
            for k in range(1,21): # skip 0 for gt
                # Example path to image label file:
                # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/aeroplane_trainval.txt
                image_label_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                    self._classes[k]+'_'+self._image_set + '.txt')
                assert os.path.exists(image_label_file), \
                    'Path does not exist: {}'.format(image_label_file)
                print("Now loading image label for class", k)
                d = {} # save to dictionary
                with open(image_label_file) as f:
                    for line in f.readlines():
                        (key, val) = line.strip().split()
                        d[key] = int(val)
                self._image_label_txt.append(d)
        gt_label = np.zeros((len(self._classes)), dtype=np.float32)
        if self.config['use_diff']:
            for k in range(1, 21):
                gt_label[k] = self._image_label_txt[k-1][index] >= 0
        else:
            for k in range(1, 21):
                # print(k, len(gt_label), len(self._image_label_txt))
                gt_label[k] = self._image_label_txt[k-1][index] > 0
        #print(index, gt_label)
        return gt_label.reshape(1, 21)


    def _load_pascal_annotation(self, index):
        """
    Load image and bounding boxes info from XML file in the PASCAL VOC
    format.
    """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            # if len(non_diff_objs) != len(objs):
            #     print 'Removed {} difficult objects'.format(
            #         len(objs) - len(non_diff_objs))
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)
        is_pseudo = np.zeros(gt_classes.shape)
        not_pseudo = np.ones(boxes.shape)
        gt_label = self._load_pascal_labels(index)
        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas,
                'pseudo': is_pseudo,
                'label': gt_label
                }
                # 'not_pseudo': not_pseudo}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._devkit_path,
            'results',
            'VOC' + self._year,
            'Main',
            filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        annopath = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric, use_diff=self.config['use_diff'])
            aps += [ap]
            print(('AP for {} = {:.4f}'.format(cls, ap)))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print(('Mean AP = {:.4f}'.format(np.mean(aps))))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print(('{:.3f}'.format(ap)))
        print(('{:.3f}'.format(np.mean(aps))))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def _do_matlab_eval(self, output_dir='output'):
        print('-----------------------------------------------------')
        print('Computing results with the official MATLAB eval code.')
        print('-----------------------------------------------------')
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
            .format(self._devkit_path, self._get_comp_id(),
                    self._image_set, output_dir)
        print(('Running:\n{}'.format(cmd)))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True


if __name__ == '__main__':
    from datasets.pascal_voc import pascal_voc

    d = pascal_voc('trainval', '2007')
    res = d.roidb
    from IPython import embed;

    embed()
