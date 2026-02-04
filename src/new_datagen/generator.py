from __future__ import annotations
from typing import Tuple
import numpy as np
import cv2      

from .config import CANVAS_SIZE
from .terraces import generate_base_surface
from .overlay_engine import apply_random_signatures


def rotate_45(img: np.ndarray, is_mask: bool = False) -> np.ndarray:

    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, 45.0, 1.0)

    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR

    rotated = cv2.warpAffine(
        img.astype(np.float32),
        M,
        (w, h),
        flags=interp,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    if is_mask:
        return rotated.astype(np.uint8)
    else:
        return np.clip(rotated, 0, 255).astype(np.uint8)



def generate_synthetic_stm_sample(
    canvas_size: Tuple[int, int] = CANVAS_SIZE,
    with_terraces: bool = True,
    with_defects: bool = True,
    with_noise: bool = False,     
    rotate_45_deg: bool = True,
    oversize_factor: float = 1.42,
    n_defects_range=(0, 6),
) -> tuple[np.ndarray, np.ndarray]:
    H, W = canvas_size
    H_big = int(H * oversize_factor)
    W_big = int(W * oversize_factor)

    img_big = generate_base_surface((H_big, W_big), add_terraces=with_terraces)
    mask_big = np.zeros_like(img_big, dtype=np.uint8)

    if with_defects:
        img_big, mask_big = apply_random_signatures(
            img_big,
            mask_big,
            n_defects_range=n_defects_range,  
        )


    if rotate_45_deg:
        img_big  = rotate_45(img_big, is_mask=False)
        mask_big = rotate_45(mask_big, is_mask=True)

    cy = H_big // 2
    cx = W_big // 2
    y0 = cy - H // 2
    y1 = y0 + H
    x0 = cx - W // 2
    x1 = x0 + W

    img  = img_big[y0:y1, x0:x1]
    mask = mask_big[y0:y1, x0:x1]

    return img, mask
