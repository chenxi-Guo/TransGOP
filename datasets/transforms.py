# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import random

from PIL import Image, ImageDraw, ImageFont
import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np
from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate


def crop(image, target, target_gaze, region):

    cropped_image = F.crop(image, *region)

    target = target.copy()
    target_gaze = target_gaze.copy()
    i, j, h, w = region

    # gaze #
    target_gaze["size"] = torch.tensor([h, w])

    if "head_box" in target_gaze:
        head_box = target_gaze["head_box"]
        head_box = head_box - torch.as_tensor([j, i, j, i])
        target_gaze["head_box"] = head_box

    if "eye" in target_gaze:
        eye = target_gaze["eye"]
        eye = eye - torch.as_tensor([j, i])
        target_gaze["eye"] = eye

    if "gaze_box" in target_gaze:
        gaze_box = target_gaze["gaze_box"]
        gaze_box = gaze_box - torch.as_tensor([j, i, j, i])
        target_gaze["gaze_box"] = gaze_box

    if "gaze_point" in target_gaze:
        gaze_point = target_gaze["gaze_point"]
        gaze_point = gaze_point - torch.as_tensor([j, i])
        target_gaze["gaze_point"] = gaze_point

    # gaze #

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])


    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")


    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target, target_gaze


def hflip(image, target, target_gaze):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    target_gaze = target_gaze.copy()

    # gaze #
    if "head_box" in target_gaze:
        head_box = target_gaze["head_box"]
        x_min = w - head_box[2]
        x_max = w - head_box[0]
        head_box[0] = x_min
        head_box[2] = x_max
        target_gaze["head_box"] = head_box

    if "eye" in target_gaze:
        eye = target_gaze["eye"]
        eye[0] = w - eye[0]
        target_gaze["eye"] = eye

    if "gaze_box" in target_gaze:
        gaze_box = target_gaze["gaze_box"]
        x_min2 = w - gaze_box[2]
        x_max2 = w - gaze_box[0]
        gaze_box[0] = x_min2
        gaze_box[2] = x_max2
        target_gaze["gaze_box"] = gaze_box

    if "gaze_point" in target_gaze:
        gaze_point = target_gaze["gaze_point"]
        gaze_point[0] = w - gaze_point[0]
        target_gaze["gaze_point"] = gaze_point
    # gaze #


    # object detection #
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target, target_gaze


def resize(image, target, target_gaze, size1, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = (size1, size1) # get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios
    h, w = size
    # gaze #
    target_gaze = target_gaze.copy()

    if "head_box" in target_gaze:
        head_box = target_gaze["head_box"]
        head_box = head_box * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target_gaze["head_box"] = head_box

    if "eye" in target_gaze:
        eye = target_gaze["eye"]
        eye = eye * torch.as_tensor([ratio_width, ratio_height])
        target_gaze["eye"] = eye

    if "gaze_box" in target_gaze:
        gaze_box = target_gaze["gaze_box"]
        gaze_box = gaze_box * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target_gaze["gaze_box"] = gaze_box

    if "gaze_point" in target_gaze:
        gaze_point = target_gaze["gaze_point"]
        gaze_point = gaze_point * torch.as_tensor([ratio_width, ratio_height])
        target_gaze["gaze_point"] = gaze_point
    target_gaze["size"] = torch.tensor([h, w])
    # gaze #


    # object detection #
    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area


    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target, target_gaze


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class ResizeDebug(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        return resize(img, target, self.size)


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict, target_gaze: dict):

        # gaze type #
        target_gaze = target_gaze.copy()
        if "head_box" in target_gaze and "gaze_point" in target_gaze:
            head_box = target_gaze["head_box"]
            gaze_point = target_gaze["gaze_point"]

            crop_x_min = min([gaze_point[0], head_box[0], head_box[2]])
            crop_y_min = min([gaze_point[1], head_box[1], head_box[3]])
            crop_x_max = max([gaze_point[0], head_box[0], head_box[2]])
            crop_y_max = max([gaze_point[1], head_box[1], head_box[3]])



            # Randomly select a random top left corner
            if crop_x_min >= 0:
                crop_x_min = random.randint(0, int(crop_x_min))
            if crop_y_min >= 0:
                crop_y_min = random.randint(0, int(crop_y_min))

            # Find the range of valid crop width and height starting from the (crop_x_min, crop_y_min)
            crop_width_min = int(crop_x_max - crop_x_min)
            crop_height_min = int(crop_y_max - crop_y_min)
            crop_width_max = int(img.width - crop_x_min)
            crop_height_max = int(img.height - crop_y_min)
            # Randomly select a width and a height
            w = random.randint(crop_width_min, crop_width_max)
            h = random.randint(crop_height_min, crop_height_max)

        # object detection type #
        # w = random.randint(self.min_size, min(img.width, self.max_size))
        # h = random.randint(self.min_size, min(img.height, self.max_size))
        region = [crop_y_min, crop_x_min, h, w]
        # region = T.RandomCrop.get_params(img, [h, w]) # output: tuple[int, int, int, int]

        return crop(img, target, target_gaze, region)



class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target, target_gaze):
        if random.random() < self.p:
            return hflip(img, target, target_gaze)
        return img, target, target_gaze


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None, target_gaze=None):
        size = random.choice(self.sizes)
        return resize(img, target, target_gaze, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target, target_gaze):
        if random.random() < self.p:
            return self.transforms1(img, target, target_gaze)
        return self.transforms2(img, target, target_gaze)


class ToTensor(object):
    def __call__(self, img, target, target_gaze):
        return F.to_tensor(img), target, target_gaze


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None, target_gaze=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        target_gaze = target_gaze.copy()
        h, w = image.shape[-2:]

        # gaze #
        if "head_box" in target_gaze:
            head_box = target_gaze["head_box"]
            head_box = box_xyxy_to_cxcywh(head_box)
            head_box = head_box / torch.tensor([w, h, w, h], dtype=torch.float32)
            target_gaze["head_box"] = head_box

        if "eye" in target_gaze:
            eye = target_gaze["eye"]
            eye = eye / torch.tensor([w, h], dtype=torch.float32)
            target_gaze["eye"] = eye

        if "gaze_box" in target_gaze:
            gaze_box = target_gaze["gaze_box"]
            gaze_box = box_xyxy_to_cxcywh(gaze_box)
            gaze_box = gaze_box / torch.tensor([w, h, w, h], dtype=torch.float32)
            target_gaze["gaze_box"] = gaze_box

        if "gaze_point" in target_gaze:
            gaze_point = target_gaze["gaze_point"]
            gaze_point = gaze_point / torch.tensor([w, h], dtype=torch.float32)
            target_gaze["gaze_point"] = gaze_point
        # gaze #


        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target, target_gaze


def get_head_box_channel(x_min, y_min, x_max, y_max, width, height, resolution, coordconv=False):
    head_box = np.array([x_min / width, y_min / height, x_max / width, y_max / height]) * resolution
    head_box = head_box.astype(int)
    head_box = np.clip(head_box, 0, resolution - 1)
    if coordconv:
        unit = np.array(range(0, resolution), dtype=np.float32)
        head_channel = []
        for i in unit:
            head_channel.append([unit + i])
        head_channel = np.squeeze(np.array(head_channel)) / float(np.max(head_channel))
        head_channel[head_box[1]:head_box[3], head_box[0]:head_box[2]] = 0
    else:
        head_channel = np.zeros((resolution, resolution), dtype=np.float32)
        head_channel[head_box[1]:head_box[3], head_box[0]:head_box[2]] = 1
    head_channel = torch.from_numpy(head_channel)
    return head_channel

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def draw_labelmap(img, pt, sigma, type='Gaussian'):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    img = to_numpy(img)

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0 or pt[0] < 0 or pt[1] < 0 or pt[0] >= img.shape[1] or pt[1] >= img.shape[0]):
        # If not, just return the image as is
        return to_torch(img)

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    try:
        img[img_y[0]:img_y[1], img_x[0]:img_x[1]] += g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    except:
        pass
    img = img / np.max(img)  # normalize heatmap so it has max value of 1
    return to_torch(img)

def preprocess_input(image):
    image /= 255.0
    return image


class gaze_postprocess(object):
    def __init__(self, input_size, output_size):

        self.input_size = input_size
        self.output_size = output_size

    def __call__(self, image, target_gaze):

        width, height = image.size
        x_min, y_min, x_max, y_max = target_gaze["head_box"]
        size = target_gaze["size"]
        gaze_x, gaze_y = target_gaze["gaze_point"]

        head_channel = get_head_box_channel(x_min, y_min, x_max, y_max, width, height,
                                            resolution=self.input_size, coordconv=False).unsqueeze(0)


        # Crop the face
        face = image.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        face = F.resize(face, (self.input_size, self.input_size))
        face = F.to_tensor(face)
        face = F.normalize(face, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # face = np.transpose(preprocess_input(np.array(face, dtype=np.float32)), (2, 0, 1)) # ?
        # image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1)) # ?

        # generate the heatmap used for deconv prediction
        gaze_heatmap = torch.zeros(self.output_size, self.output_size)  # set the size of the output
        gaze_heatmap = draw_labelmap(gaze_heatmap, [int(gaze_x / width * self.output_size),
                                                    int(gaze_y / height * self.output_size)], 6, type='Gaussian')

        # face = np.array(face, dtype=np.float32)
        # image = np.array(image, dtype=np.float32)
        # head_channel = np.array(head_channel, dtype=np.float32)
        # gaze_heatmap = np.array(gaze_heatmap, dtype=np.float32)


        return image, face, head_channel, gaze_heatmap, target_gaze




class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, target_gaze, face=None, head_channel=None, gaze_heatmap=None):

        for t in self.transforms:

            n = type(t).__name__
            if n == 'gaze_postprocess':
                image, face, head_channel, gaze_heatmap, target_gaze = t(image, target_gaze)
            else:
                image, target, target_gaze = t(image, target, target_gaze)

        return image, target, target_gaze, face, head_channel, gaze_heatmap

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
