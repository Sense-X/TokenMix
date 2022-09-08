""" Image dataset with label maps
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data

import os
import re
import io
import torch
import random
import tarfile
import numpy as np
import logging
from PIL import Image
from imagenet_dataset import ImageNetDataset, pil_loader
logger = logging.getLogger('token_label_dataset')

IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg']


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


class DatasetTokenLabel(ImageNetDataset):

    def __init__(
            self, 
            root_dir, 
            meta_file, 
            label_root,
            transform=None,
            use_ceph=False):

        self.label_root = label_root
        self.transform = transform
        super().__init__(root_dir, meta_file, transform, use_ceph=use_ceph)

    def __getitem__(self, idx):
        try:
            img_path = str(self.metas_names[idx], encoding='utf-8')
            score_path = os.path.join(
                self.label_root,
                '/'.join(img_path.split('/')[-2:]).split('.')[0] + '.pt')

            label = self.metas_labels[idx]
            if self.use_ceph:
                value = self.mclient.Get(img_path)
                img_bytes = np.fromstring(value, np.uint8)
                buff = io.BytesIO(img_bytes)
                with Image.open(buff) as img:
                    img = img.convert('RGB')

                score_value = self.mclient.Get(score_path)
                buffer = io.BytesIO(score_value)
                score_maps = torch.load(buffer, 'cpu').float()
            else:
                img = pil_loader(img_path)
                score_maps = torch.load(score_path).float()

            if self.transform is not None:
                img, score_maps = self.transform(img, score_maps)
            # append ground truth after coords
            score_maps[-1,0,0,5] = label
            return img, score_maps
        except Exception as e:
            logger.info(f'Error when load {idx}')
            logger.info(e)
            return self.__getitem__(random.randint(0, len(self.metas_names) - 1))
