import os
import json
import numpy as np

from .detection import DETECTION
from ..paths import get_file_path


class TERRORPOST(DETECTION):
    def __init__(self, db_config, split=None, sys_config=None):
        assert split is None or sys_config is not None
        super(TERRORPOST, self).__init__(db_config)

        self._mean    = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
        self._std     = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        self._terrorpost_cls_ids = list(np.array(list(range(23))) + 1)

        self._terrorpost_cls_names = [
            'BK_LOGO',
            'guns_anime',
            'zhongguojinwen_logo',
            'knives_kitchen',
            'idcard_positive',
            'bankcard_positive',
            'isis_flag',
            'guns_true',
            'falungong_logo',
            'nazi_logo',
            'islamic_flag',
            'tibetan_flag',
            'knives_false',
            'guns_tools',
            'china_guoqi_flag',
            'bankcard_negative',
            'mingjing_logo',
            'gongzhang_logo',
            'taiwan_bairiqi_flag',
            'not_terror_card_text',
            'not_terror',
            'idcard_negative',
            'knives_true'
        ]  # 23 class

        self._cls2terrorpost = {ind + 1: terrorpost_id for ind, terrorpost_id in enumerate(self._terrorpost_cls_ids)}
        self._terrorpost2cls = {terrorpost_id: cls_id for cls_id, terrorpost_id in self._cls2terrorpost.items()}

        self._terrorpost2name = {cls_id: cls_name for cls_id, cls_name in zip(self._terrorpost_cls_ids, self._terrorpost_cls_names)}
        self._name2terrorpost = {cls_name: cls_id for cls_name, cls_id in self._terrorpost2name.items()}

        if split is not None:
            terrorpost_dir = os.path.join(sys_config.data_dir, 'terrorpost')

            self._split = {
                # 'trainval': 'trainval2018',
                'wa_v1.3.01_trainval': 'wa_v1.3.01_trainval',
                'wa_v1.3.01_test': 'wa_v1.3.01_test',
                'testdev': 'testdev2018'
            }[split]

            # self._data_dir = os.path.join(terrorpost_dir, 'images', self._split)
            self._data_dir = os.path.join(terrorpost_dir, 'images')
            # self._anno_file = os.path.join(terrorpost_dir, 'annotations', 'instances_{}.json'.format(self._split))
            self._anno_file = os.path.join(terrorpost_dir, 'annotations', '{}.json'.format(self._split))

            self._detections, self._eval_ids = self._load_terrorpost_annos()
            self._image_ids = list(self._detections.keys())
            self._db_inds = np.arange(len(self._image_ids))

    def _load_terrorpost_annos(self):
        from pycocotools.coco import COCO

        terrorpost = COCO(self._anno_file)
        self._terrorpost = terrorpost

        class_ids = terrorpost.getCatIds()
        image_ids = terrorpost.getImgIds()
        print("class_ids: {}".format(list(set(class_ids))))
        eval_ids = {}
        detections = {}
        for image_id in image_ids:
            image = terrorpost.loadImgs(image_id)[0]
            dets = []

            eval_ids[image['file_name']] = image_id
            for class_id in class_ids:
                annotation_ids = terrorpost.getAnnIds(imgIds=image['id'], catIds=class_id)
                annotations = terrorpost.loadAnns(annotation_ids)
                category = self._terrorpost2cls[class_id]
                for annotation in annotations:
                    det = annotation['bbox'] + [category]
                    det[2] += det[0]
                    det[3] += det[1]
                    dets.append(det)

            file_name = image['file_name']
            if len(dets) == 0:
                detections[file_name] = np.zeros((0, 5), dtype=np.float32)
            else:
                detections[file_name] = np.array(dets, dtype=np.float32)
        return detections, eval_ids

    def image_path(self, ind):
        if self._data_dir is None:
            raise ValueError('Data directory is not set')

        db_ind = self._db_inds[ind]
        file_name = self._image_ids[db_ind]
        return os.path.join(self._data_dir, file_name)

    def detections(self, ind):
        db_ind = self._db_inds[ind]
        file_name = self._image_ids[db_ind]
        return self._detections[file_name].copy()

    def cls2name(self, cls):
        terrorpost = self._cls2terrorpost[cls]
        return self._terrorpost2name[terrorpost]

    def _to_float(self, x):
        return float('{:.2f}'.format(x))

    def convert_to_coco(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            coco_id = self._eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._cls2terrorpost[cls_ind]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]

                    score = bbox[4]
                    bbox = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        'image_id': int(coco_id),
                        'category_id': int(category_id),
                        'bbox': bbox,
                        'score': float('{:.2f}'.format(score))
                    }

                    detections.append(detection)
        return detections

    def evaluate(self, result_json, cls_ids, image_ids):
        from pycocotools.cocoeval import COCOeval

        if self._split == 'testdev':
            return None

        terrorpost = self._terrorpost

        eval_ids = [self._eval_ids[image_id] for image_id in image_ids]
        cat_ids = [self._cls2terrorpost[cls_id] for cls_id in cls_ids]

        terrorpost_dets = terrorpost.loadRes(result_json)
        terrorpost_eval = COCOeval(terrorpost, terrorpost_dets, 'bbox')
        terrorpost_eval.params.imgIds = eval_ids
        terrorpost_eval.params.catIds = cat_ids
        terrorpost_eval.evaluate()
        terrorpost_eval.accumulate()
        terrorpost_eval.summarize()
        return terrorpost_eval.stats[0], terrorpost_eval.stats[12:]