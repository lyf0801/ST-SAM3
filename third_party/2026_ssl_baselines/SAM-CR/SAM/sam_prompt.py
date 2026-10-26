import cv2
import numpy as np
import time

from PIL import Image
from tqdm import tqdm
import torch
import os
import shutil
from SAM.segment_anything import SamPredictor, sam_model_registry

def is_point_inside_mask(mask, point):
    x, y = int(point[0]), int(point[1])
    if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
        return mask[y, x] == 255
    return False

def find_nearest_inner_point(contour, mask, max_search_steps=20):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    if is_point_inside_mask(mask, (cx, cy)):
        return (cx, cy)

    rect = cv2.minAreaRect(contour)
    center, size, angle = rect
    angle_rad = np.deg2rad(angle)

    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)

    for step in range(1, max_search_steps):
        for sign in [-1, 1]:
            x = int(cx + sign * step * dx)
            y = int(cy + sign * step * dy)
            if is_point_inside_mask(mask, (x, y)):
                return (x, y)
    return None


def load_mask_and_safe_centers(binary_mask):
    if binary_mask.dtype == np.float32:
        binary_mask = (binary_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    safe_centers = []
    bounding_boxes = []
    for contour in contours:
        if len(contour) < 5:
            continue
        point = find_nearest_inner_point(contour, binary_mask)
        if point is not None:
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, x+w, y+h))
            safe_centers.append(point)

    return np.array(safe_centers),np.array(bounding_boxes)