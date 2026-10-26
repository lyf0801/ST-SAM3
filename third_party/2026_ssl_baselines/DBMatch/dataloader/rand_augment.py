# code in this file is adpated from
# https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/third_party/auto_augment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/libml/ctaugment.py
import logging
import random

import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image
import colorsys
import cv2

from skimage.exposure import match_histograms


def color_swap(img, prob=0.5):
    if random.random() < prob:
        order = [0,1,2]
        random.shuffle(order)
        channels = img.split()    
        r,g,b = channels[order[0]], channels[order[1]], channels[order[2]]
        img = Image.merge('RGB', (r,g,b))
    return img


def hsv_shift(img, prob=0.5):
    if random.random() < prob:
        img = np.asarray(img)

        h_range, s_range, v_range = 20, 30, 20
        h_shift = random.randint(-h_range, h_range)
        s_shift = random.randint(-s_range, s_range)
        v_shift = random.randint(-v_range, v_range)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hue, sat, val =cv2.split(img)

        lut_hue = np.arange(0, 256, dtype=np.int16)
        lut_hue = np.mod(lut_hue+h_shift, 180).astype(np.uint8)
        hue = cv2.LUT(hue, lut_hue)

        lut_sat = np.arange(0, 256, dtype=np.int16)
        lut_sat = np.clip(lut_sat+s_shift, 0, 255).astype(np.uint8)
        sat = cv2.LUT(sat, lut_sat)

        lut_val = np.arange(0, 256, dtype=np.int16)
        lut_val = np.clip(lut_val+v_shift, 0, 255).astype(np.uint8)
        val = cv2.LUT(val, lut_val)

        img = cv2.merge((hue, sat, val)).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        img = Image.fromarray(img)
    return img

def horizontal_flip(image, label, prob=0.5):
    if random.random() < prob:
        image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        label = label.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    return image, label

def vertical_flip(image, label, prob=0.5):
    if random.random() < prob:
        image = image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        label = label.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    return image, label

def rotate(image, label):
    angle = random.choice([90, 180, 270, 360])
    image = image.rotate(angle, expand=True)
    label = label.rotate(angle, expand=True)
    return image, label

def random_resize_and_crop(image, label, ignore_index=-1):
    # 将 PIL 图像转换为 NumPy 数组
    image_np = np.array(image)
    label_np = np.array(label)

    # 获取原始尺寸
    org_height, org_width = image_np.shape[:2]
    scale = random.uniform(0.5, 2.0)

    # 计算新的尺寸
    new_width = int(org_width * scale)
    new_height = int(org_height * scale)

    # 缩放图像和标签
    resized_image = Image.fromarray(image_np).resize((new_width, new_height), Image.BILINEAR)
    resized_label = Image.fromarray(label_np).resize((new_width, new_height), Image.NEAREST)
    resized_image = np.array(resized_image)
    resized_label = np.array(resized_label, dtype=np.int32)

    # 根据 scale 大小处理图像和标签
    if scale < 1:
        # 计算填充尺寸以恢复到原始尺寸
        left, right = 0, org_width
        top, bottom = 0, org_height
        pad_width_left = (org_width - new_width) // 2
        pad_width_right = org_width - new_width - pad_width_left
        pad_height_top = (org_height - new_height) // 2
        pad_height_bottom = org_height - new_height - pad_height_top
        resized_image = np.pad(resized_image, 
                               ((pad_height_top, pad_height_bottom), 
                                (pad_width_left, pad_width_right), (0, 0)), 
                               mode='constant', constant_values=0)
        resized_label = np.pad(resized_label, 
                               ((pad_height_top, pad_height_bottom), 
                                (pad_width_left, pad_width_right)), 
                                mode='constant', constant_values=ignore_index)

        # 将整个填充图像转换回 PIL 图像
        cropped_image = Image.fromarray(resized_image)
        cropped_label = resized_label
    else:
        # 随机选择裁剪位置
        left = random.randint(0, new_width - org_width)
        right = left + org_width
        top = random.randint(0, new_height - org_height)
        bottom = top + org_height
        cropped_image = resized_image[top:bottom, left:right]
        cropped_label = resized_label[top:bottom, left:right]

        # 将裁剪后的图像转换回 PIL 图像
        cropped_image = Image.fromarray(cropped_image)
        cropped_label = cropped_label
    return cropped_image, cropped_label





def adjust_color_hue(img, max_temp_change):
    """
    随机调整输入 PIL 图像的色温。

    参数:
        img (PIL.Image): 输入的 RGB 图像。
        max_temp_change (float): 色温变化范围（正负），取值范围在 [-0.5, 0.5] 之间。

    返回:
        PIL.Image: 调整色温后的图像。
    """
    def rgb_to_hls(img_np):
        return np.apply_along_axis(lambda x: np.array(colorsys.rgb_to_hls(*x)), -1, img_np)

    def hls_to_rgb(img_hls):
        return np.apply_along_axis(lambda x: np.array(colorsys.hls_to_rgb(*x)), -1, img_hls)

    if not isinstance(img, Image.Image):
        raise TypeError("输入图像必须是 PIL 图像")

    if not (-0.5 <= max_temp_change <= 0.5):
        raise ValueError("max_temp_change 必须在 [-0.5, 0.5] 范围内")

    # 转换为 NumPy 数组
    img_np = np.array(img) / 255.0  # 归一化为 [0, 1]
    hue_factor = np.random.uniform(-max_temp_change, max_temp_change)

    # 将 RGB 转换为 HLS
    img_hls = rgb_to_hls(img_np)
    
    # 调整色调
    img_hls[..., 0] = (img_hls[..., 0] + hue_factor) % 1.0

    # 将 HLS 转换回 RGB
    img_rgb = hls_to_rgb(img_hls)
    
    # 转换为 PIL 图像
    adjusted_img = Image.fromarray((img_rgb * 255).astype(np.uint8))
    
    return adjusted_img





PARAMETER_MAX = 10

def ColorSwap(img, **kwarg):
    order = [0,1,2]
    random.shuffle(order)
    channels = img.split()    
    r,g,b = channels[order[0]], channels[order[1]], channels[order[2]]
    img = Image.merge('RGB', (r,g,b))
    return img

def HsvShift(img, **kwarg):
    img = np.asarray(img)

    h_range, s_range, v_range = 20, 30, 20
    h_shift = random.randint(-h_range, h_range)
    s_shift = random.randint(-s_range, s_range)
    v_shift = random.randint(-v_range, v_range)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue, sat, val =cv2.split(img)

    lut_hue = np.arange(0, 256, dtype=np.int16)
    lut_hue = np.mod(lut_hue+h_shift, 180).astype(np.uint8)
    hue = cv2.LUT(hue, lut_hue)

    lut_sat = np.arange(0, 256, dtype=np.int16)
    lut_sat = np.clip(lut_sat+s_shift, 0, 255).astype(np.uint8)
    sat = cv2.LUT(sat, lut_sat)

    lut_val = np.arange(0, 256, dtype=np.int16)
    lut_val = np.clip(lut_val+v_shift, 0, 255).astype(np.uint8)
    val = cv2.LUT(val, lut_val)

    img = cv2.merge((hue, sat, val)).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    img = Image.fromarray(img)
    return img

def AutoContrast(img, **kwarg):
    return PIL.ImageOps.autocontrast(img)

def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v)

def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v)

def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)

def Cutout(img, v, max_v, bias=0):
    if v == 0:
        return img
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(img.size))
    return CutoutAbs(img, v)

def CutoutAbs(img, v, **kwarg):
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # gray
    color = (127, 127, 127)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img

def Equalize(img, **kwarg):
    return PIL.ImageOps.equalize(img)

def Identity(img, **kwarg):
    return img

def Invert(img, **kwarg):
    return PIL.ImageOps.invert(img)

def Posterize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v)

def Rotate(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v)

def Sharpness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)

def ShearX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))

def ShearY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))

def Solarize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.solarize(img, 256 - v)

def SolarizeAdd(img, v, max_v, bias=0, threshold=128):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)

def TranslateX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX

def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)


def fixmatch_augment_pool(no_swap=False):
    # FixMatch paper
    if no_swap:
        augs = [(AutoContrast, None, None),
                (Brightness, 0.9, 0.05),
                (Color, 0.9, 0.05),
                (Contrast, 0.9, 0.05),
                (Equalize, None, None),
                (Identity, None, None),
                (Posterize, 4, 4),
                (Sharpness, 0.9, 0.05),
                (Solarize, 256, 0),
                ]
    else:
        augs = [(AutoContrast, None, None),
                (Brightness, 0.9, 0.05),
                (Color, 0.9, 0.05),
                (Contrast, 0.9, 0.05),
                (Equalize, None, None),
                (Identity, None, None),
                (ColorSwap, None, None),
                (HsvShift, None, None),
                (Posterize, 4, 4),
                (Sharpness, 0.9, 0.05),
                (Solarize, 256, 0),
                ]
    return augs


def my_augment_pool():
    # Test
    augs = [(AutoContrast, None, None),
            (Brightness, 1.8, 0.1),
            (Color, 1.8, 0.1),
            (Contrast, 1.8, 0.1),
            (Cutout, 0.2, 0),
            (Equalize, None, None),
            (Invert, None, None),
            (Posterize, 4, 4),
            # (Rotate, 30, 0),
            (Sharpness, 1.8, 0.1),
            # (ShearX, 0.3, 0),
            # (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (SolarizeAdd, 110, 0),
            # (TranslateX, 0.45, 0),
            # (TranslateY, 0.45, 0)
            ]
    return augs


class RandAugmentMC(object):
    def __init__(self, n, m, no_swap=False):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool(no_swap)

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
        img = CutoutAbs(img, 16)
        return img
