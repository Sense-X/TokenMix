""" Mixup and Cutmix

Papers:
mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)

CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features (https://arxiv.org/abs/1905.04899)

Code Reference:
CutMix: https://github.com/clovaai/CutMix-PyTorch
ReLabel: https://github.com/naver-ai/relabel_imagenet
TokenLabeling: https://github.com/zihangJiang/TokenLabeling


"""
import math
import random
import cv2
import numpy as np
import torch

from torchvision.ops import roi_align


def get_featuremaps(label_maps_topk, num_classes, device='cuda'):
    label_maps_topk_sizes = label_maps_topk[0].size()
    label_maps = torch.full([label_maps_topk.size(0), num_classes, label_maps_topk_sizes[2],
                              label_maps_topk_sizes[3]], 0, dtype=torch.float32 ,device=device)
    for _label_map, _label_topk in zip(label_maps, label_maps_topk):
        _label_map = _label_map.scatter_(
            0,
            _label_topk[1][:, :, :].long(),
            _label_topk[0][:, :, :].float()
        )
    return label_maps

def get_label(label_maps, batch_coords,label_size=1,device='cuda'):
    num_batches = label_maps.size(0)
    target_label = roi_align(
        input=label_maps,
        boxes=torch.cat(
            [torch.arange(num_batches).view(num_batches,
                                            1).float().to(device),
             batch_coords.float() * label_maps.size(3) - 0.5], 1),
        output_size=(label_size, label_size))
    return target_label

def get_labelmaps_with_coords(label_maps_topk, num_classes, on_value=1., off_value=0.,label_size=1, device='cuda'):
    random_crop_coords = label_maps_topk[:,2,0,0,:4].view(-1, 4)
    random_crop_coords[:, 2:] += random_crop_coords[:, :2]
    random_crop_coords = random_crop_coords.to(device)

    # get full label maps from raw topk labels
    # b, 1000, h, w
    label_maps = get_featuremaps(label_maps_topk=label_maps_topk,
                               num_classes=num_classes,device=device)

    # get token-level label and ground truth
    token_label = get_label(label_maps=label_maps,
                          batch_coords=random_crop_coords,
                          label_size=label_size,
                          device=device)
    B,C = token_label.shape[:2]
    token_label = token_label*on_value+off_value
    # output: B, 1000, H, W
    return token_label


def one_hot(x, num_classes, on_value=1., off_value=0., device='cuda'):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)


def mixup_target(target, num_classes, lam=1., smoothing=0.0, device='cuda'):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value, device=device)
    y2 = one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value, device=device)
    return y1 * lam + y2 * (1. - lam)


def rand_bbox(img_shape, lam, margin=0., count=None):
    """ Standard CutMix bounding-box
    Generates a random square bbox based on lambda value. This impl includes
    support for enforcing a border margin as percent of bbox dimensions.

    Args:
        img_shape (tuple): Image shape as tuple
        lam (float): Cutmix lambda value
        margin (float): Percentage of bbox dimension to enforce as margin (reduce amount of box outside image)
        count (int): Number of bbox to generate
    """
    ratio = np.sqrt(1 - lam)
    img_h, img_w = img_shape[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
    cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
    cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
    yl = np.clip(cy - cut_h // 2, 0, img_h)
    yh = np.clip(cy + cut_h // 2, 0, img_h)
    xl = np.clip(cx - cut_w // 2, 0, img_w)
    xh = np.clip(cx + cut_w // 2, 0, img_w)
    return yl, yh, xl, xh


def rand_bbox_minmax(img_shape, minmax, count=None):
    """ Min-Max CutMix bounding-box
    Inspired by Darknet cutmix impl, generates a random rectangular bbox
    based on min/max percent values applied to each dimension of the input image.

    Typical defaults for minmax are usually in the  .2-.3 for min and .8-.9 range for max.

    Args:
        img_shape (tuple): Image shape as tuple
        minmax (tuple or list): Min and max bbox ratios (as percent of image size)
        count (int): Number of bbox to generate
    """
    assert len(minmax) == 2
    img_h, img_w = img_shape[-2:]
    cut_h = np.random.randint(int(img_h * minmax[0]), int(img_h * minmax[1]), size=count)
    cut_w = np.random.randint(int(img_w * minmax[0]), int(img_w * minmax[1]), size=count)
    yl = np.random.randint(0, img_h - cut_h, size=count)
    xl = np.random.randint(0, img_w - cut_w, size=count)
    yu = yl + cut_h
    xu = xl + cut_w
    return yl, yu, xl, xu

def saliency_bbox(img, lam):
    size = img.size()
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # initialize OpenCV's static fine grained saliency detector and compute the saliency map
    temp_img = img.cpu().numpy().transpose(1, 2, 0)
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(temp_img)
    saliencyMap = (saliencyMap * 255).astype("uint8")

    maximum_indices = np.unravel_index(np.argmax(saliencyMap, axis=None), saliencyMap.shape)
    x = maximum_indices[0]
    y = maximum_indices[1]

    bbx1 = np.clip(x - cut_w // 2, 0, W)
    bby1 = np.clip(y - cut_h // 2, 0, H)
    bbx2 = np.clip(x + cut_w // 2, 0, W)
    bby2 = np.clip(y + cut_h // 2, 0, H)

    return bby1, bby2, bbx1, bbx2
    # return bbx1, bby1, bbx2, bby2


def cutmix_bbox_and_lam(img_shape, lam, ratio_minmax=None, correct_lam=True, count=None):
    """ Generate bbox and apply lambda correction.
    """
    if ratio_minmax is not None:
        yl, yu, xl, xu = rand_bbox_minmax(img_shape, ratio_minmax, count=count)
    else:
        yl, yu, xl, xu = rand_bbox(img_shape, lam, count=count)
    if correct_lam or ratio_minmax is not None:
        bbox_area = (yu - yl) * (xu - xl)
        lam = 1. - bbox_area / float(img_shape[-2] * img_shape[-1])
    return (yl, yu, xl, xu), lam

def generate_mask(lam, device, mask_token_num_start, min_num_patches=1):
    width = 14
    height = 14
    min_aspect = 0.3
    log_aspect_ratio = (math.log(min_aspect), math.log(1 / min_aspect))
    mask = np.zeros(shape=(14, 14), dtype=np.int)
    mask_ratio = 1 - lam
    # num_masking_patches = random.uniform(min_num_patches, max_num_patches)
    num_masking_patches = min(width * height, int(mask_ratio * width * height) + mask_token_num_start)
    # min_num_patches = 1
    max_num_patches = width * height
    mask_count = 0

    while mask_count < num_masking_patches:
        max_mask_patches = num_masking_patches - mask_count
        max_mask_patches = min(max_mask_patches, max_num_patches)
        delta = 0
        for attempt in range(10):
            target_area = random.uniform(min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < width and h < height:
                top = random.randint(0, height - h)
                left = random.randint(0, width - w)

                num_masked = mask[top: top + h, left: left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        if delta == 0:
            break
        else:
            mask_count += delta
    mask = torch.from_numpy(mask).float().to(device).unsqueeze(0).unsqueeze(0)
    lam = 1 - mask_count / max_num_patches
    return mask, lam


def generate_mask_random(lam, device, mask_token_num_start=14):
    width = 14
    height = 14
    min_aspect = 0.3
    log_aspect_ratio = (math.log(min_aspect), math.log(1 / min_aspect))
    mask = np.zeros(shape=(14 * 14), dtype=np.int)
    mask_ratio = 1 - lam
    # num_masking_patches = random.uniform(min_num_patches, max_num_patches)
    # num_masking_patches = int(mask_ratio * width * height)
    num_masking_patches = min(width * height, int(mask_ratio * width * height) + mask_token_num_start)

    mask_idx = np.random.permutation(14 * 14)[:num_masking_patches]
    mask[mask_idx] = 1
    mask = mask.reshape(14, 14)

    mask = torch.from_numpy(mask).float().to(device).unsqueeze(0).unsqueeze(0)
    lam = 1 - num_masking_patches / (14 * 14)
    return mask, lam


class Mixup:
    """ Mixup/Cutmix that applies different params to each element or whole batch

    Args:
        mixup_alpha (float): mixup alpha value, mixup is active if > 0.
        cutmix_alpha (float): cutmix alpha value, cutmix is active if > 0.
        cutmix_minmax (List[float]): cutmix min/max image ratio, cutmix is active and uses this vs alpha if not None.
        prob (float): probability of applying mixup or cutmix per batch or element
        switch_prob (float): probability of switching to cutmix instead of mixup when both are active
        mode (str): how to apply mixup/cutmix params (per 'batch', 'pair' (pair of elements), 'elem' (element)
        correct_lam (bool): apply lambda correction when cutmix bbox clipped by image borders
        label_smoothing (float): apply label smoothing to the mixed target tensor
        num_classes (int): number of classes for target
    """
    def __init__(self, mixup_alpha=1., cutmix_alpha=0., cutmix_minmax=None, prob=1.0, switch_prob=0.5,
                 mode='batch', correct_lam=True, label_smoothing=0.1, num_classes=1000,
                 mask_type='block', minimum_tokens=14):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        if self.cutmix_minmax is not None:
            assert len(self.cutmix_minmax) == 2
            # force cutmix alpha == 1.0 when minmax active to keep logic simple & safe
            self.cutmix_alpha = 1.0
        self.mix_prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.mode = mode
        self.correct_lam = correct_lam  # correct lambda based on clipped area for cutmix
        self.mixup_enabled = True  # set to false to disable mixing (intended tp be set by train loop)
        self.mask_type = mask_type
        self.minimum_tokens = minimum_tokens
        self.lam_constant = 0.5

    def _params_per_elem(self, batch_size):
        lam = np.ones(batch_size, dtype=np.float32)
        use_cutmix = np.zeros(batch_size, dtype=np.bool)
        if self.mixup_enabled:
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                use_cutmix = np.random.rand(batch_size) < self.switch_prob
                lam_mix = np.where(
                    use_cutmix,
                    np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size),
                    np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size))
            elif self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size)
            elif self.cutmix_alpha > 0.:
                use_cutmix = np.ones(batch_size, dtype=np.bool)
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = np.where(np.random.rand(batch_size) < self.mix_prob, lam_mix.astype(np.float32), lam)
        return lam, use_cutmix

    def _params_per_batch(self):
        lam = 1.
        use_cutmix = False
        if self.mixup_enabled and np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                use_cutmix = np.random.rand() < self.switch_prob
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha) if use_cutmix else \
                    np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.cutmix_alpha > 0.:
                use_cutmix = True
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = float(lam_mix)
        return lam, use_cutmix

    def _mix_batch(self, x):
        lam, use_cutmix = self._params_per_batch()
        # if lam == 1.:
        #     return 1., None
        mask = None
        if use_cutmix:
            lam = self.lam_constant
            if self.mask_type == 'block':
                mask, lam = generate_mask(lam, x.device, self.minimum_tokens)
            elif self.mask_type == 'random':
                mask, lam = generate_mask_random(lam, x.device, self.minimum_tokens)
            else:
                raise ValueError(f"unsupported mask type {self.mask_type}")

            mask_224 = torch.nn.functional.interpolate(mask, size=(224, 224), mode='nearest')

            x_flip = x.flip(0).mul_(mask_224)
            x.mul_(1 - mask_224).add_(x_flip)

        else:
            x_flipped = x.flip(0).mul_(1. - lam)
            x.mul_(lam).add_(x_flipped)
        return lam, mask

    @torch.no_grad()
    def __call__(self, x, target):
        assert len(x) % 2 == 0, 'Batch size should be even when using this'
        lam, mask = self._mix_batch(x)
        if mask is not None:
            label_map = get_labelmaps_with_coords(target, self.num_classes, label_size=14)
            hard_label = target[:, 2, 0, 0, 5].view(-1).to(dtype=torch.int64)
            B = len(hard_label)
            y1_nonmask = label_map[torch.arange(B), hard_label, :, :]
            y1_nonmask = y1_nonmask / (y1_nonmask.sum((1, 2), keepdim=True) + 1e-8)

            y2_nonmask = y1_nonmask.flip(0)
            mask.squeeze_()
            y1_masked = y1_nonmask * (1 - mask)
            y2_masked = y2_nonmask * mask
            y1 = y1_masked.sum((1, 2))
            y2 = y2_masked.sum((1, 2))

            off_value = self.label_smoothing / self.num_classes
            on_value1 = y1 - self.label_smoothing / 2 + off_value / 2
            on_value2 = y2 - self.label_smoothing / 2 + off_value / 2
            y1 = one_hot(hard_label, self.num_classes, on_value=on_value1.view(-1, 1), off_value=off_value / 2, device=x.device)
            y2 = one_hot(hard_label.flip(0), self.num_classes, on_value=on_value2.view(-1, 1), off_value=off_value / 2, device=x.device)
            # target = y1 * lam + y2 * (1. - lam)
            target = y1 + y2
            return x, target
        else:
            hard_label = target[:, 2, 0, 0, 5].view(-1).to(dtype=torch.int64)
            target = mixup_target(hard_label, self.num_classes, lam, self.label_smoothing)
            return x, target



