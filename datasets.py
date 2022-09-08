import os
import json
import random
import numpy as np
from PIL import ImageDraw

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, \
    IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform
from timm.data.transforms import str_to_interp_mode

from imagenet_dataset import ImageNetDataset


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        # root = os.path.join(args.data_path, 'train' if is_train else 'val')
        # dataset = datasets.ImageFolder(root, transform=transform)
        if is_train:
            dataset = ImageNetDataset(
                root_dir=args.root_dir_train,
                meta_file=args.meta_file_train,
                transform=transform,
            )
        else:
            dataset = ImageNetDataset(
                root_dir=args.root_dir_val,
                meta_file=args.meta_file_val,
                transform=transform,
            )
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    scale = getattr(args, 'scale', None)
    imagenet_default_mean_and_std = getattr(args, 'imagenet_default_mean_and_std', True)
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            scale=scale,
            mean=IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    test_size = args.input_size
    crop = test_size < 320
    test_interpolation = str_to_interp_mode(getattr(args, 'test_interpolation', 'bicubic'))
    if resize_im:
        if crop:
            size = int((256 / 224) * test_size)
            t.append(
                transforms.Resize(size, interpolation=test_interpolation),  # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(test_size))
        else:
            t.append(
                transforms.Resize((test_size,test_size), interpolation=test_interpolation),  # to maintain same ratio w.r.t. 224 images
            )

    t.append(transforms.ToTensor())
    # t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    if imagenet_default_mean_and_std:
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    else:
        t.append(transforms.Normalize(IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD))
    return transforms.Compose(t)



def build_dataset_relabel(args):
    from relabel.label_transforms_factory import create_token_label_transform
    from relabel.dataset import DatasetTokenLabel

    scale = getattr(args, 'scale', None)
    transform = create_token_label_transform(
        input_size=args.input_size,
        is_training=True,
        use_prefetcher=False,
        no_aug=False,
        scale=scale,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        interpolation=args.train_interpolation,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_num_splits=0,
        crop_pct=None,
        tf_preprocessing=False,
        separate=False,
    )

    dataset = DatasetTokenLabel(args.root_dir_train, args.meta_file_train, args.label_dir)
    dataset.transform = transform

    nb_classes = 1000
    return dataset, nb_classes