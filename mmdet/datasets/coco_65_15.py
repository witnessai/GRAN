import numpy as np
from pycocotools.coco import COCO

from .custom import CustomDataset
from .registry import DATASETS


@DATASETS.register_module
class CocoDataset_65_15(CustomDataset):

    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'boat',
               'traffic_light', 'fire_hydrant',
               'stop_sign', 'bench', 'bird', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie',
               'skis', 'sports_ball', 'kite', 'baseball_bat',
               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
               'bottle', 'wine_glass', 'cup', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'orange', 'broccoli', 'carrot',
               'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'tv', 'laptop',
               'remote', 'keyboard', 'cell_phone', 'microwave',
               'oven', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy_bear', 'toothbrush',#
               'airplane', 'train', 'parking meter', 'cat', 'bear', 'suitcase', 'frisbee', 'snowboard', 'fork',
               'sandwich', 'hot dog', 'toilet', 'mouse', 'toaster', 'hair drier')


    def load_annotations(self, ann_file):
        self.unseen_class_ids = (5, 7, 14, 17, 23, 33, 34, 36, 48, 54, 58, 70, 74, 80, 89)
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.seen_cat2label = {}
        index = 1
        for cat_id in self.cat_ids:
            if cat_id not in self.unseen_class_ids:
                self.seen_cat2label[cat_id] = index
                index+=1
        self.unseen_cat2label = {
            cat_id: i + 1 + 65
            for i, cat_id in enumerate(self.unseen_class_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        # print(img_infos[0])
        # import ipdb 
        # ipdb.set_trace()
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(self.img_infos[idx], ann_info)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                if ann['category_id'] in self.seen_cat2label:
                    gt_labels.append(self.seen_cat2label[ann['category_id']])
                else:
                    gt_labels.append(self.unseen_cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann
