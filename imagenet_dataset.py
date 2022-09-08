import os.path as osp
import numpy as np
import io
from PIL import Image
import logging
import random
from torch.utils.data import Dataset
logger = logging.getLogger('global')

def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

class ImageNetDataset(Dataset):
    def __init__(self, root_dir, meta_file, transform=None, use_ceph=False):

        self.root_dir = root_dir
        self.meta_file = meta_file
        self.transform = transform
        self.initialized = False
        self.use_ceph = use_ceph

        with open(meta_file) as f:
            lines = f.readlines()
        self.num = len(lines)
        metas_names = []
        metas_labels = []
        for line in lines:
            filename, label = line.rstrip().split()
            metas_names.append(osp.join(self.root_dir, filename))
            metas_labels.append(int(label))
        self.metas_names = np.string_(metas_names)
        self.metas_labels = np.int_(metas_labels)
        self.initialized = False
        if self.use_ceph:
            from petrel_client.client import Client as CephClient
            self.mclient = CephClient()
            self.initialized = True

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        try:
            img_path = str(self.metas_names[idx], encoding='utf-8')
            label = self.metas_labels[idx]
            if self.use_ceph:
                value = self.mclient.Get(img_path)
                img_bytes = np.fromstring(value, np.uint8)
                buff = io.BytesIO(img_bytes)
                with Image.open(buff) as img:
                    img = img.convert('RGB')
            else:
                img = pil_loader(img_path)

            if self.transform is not None:
                img = self.transform(img)

            return img, label
        except Exception as e:
            logger.info(f'Error when load {idx}')
            logger.info(e)
            return self.__getitem__(random.randint(0, len(self.metas_names) - 1))
            