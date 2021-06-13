from logging import getLogger
from typing import List

import torch
import torch.nn as nn
from torchvision import transforms

from ..mean_std import get_mean, get_std

logger = getLogger(__name__)


def get_spatial_transform(
    crop_type: str,
    size: int,
    hflip: bool,
    vflip: bool,
    colorjitter: bool,
    crop_min_scale: float = 0.25,
    crop_min_ratio: float = 0.75,
    scales: List[float] = [1.0],
    scale_step: float = 1 / (2 ** (1 / 4)),
    brightness: float = 0,
    contrast: float = 0,
    saturation: float = 0,
    hue: float = 0,
    device: str = "cpu",
) -> nn.Sequential:
    logger.info("Setting up spatial transform...")

    spatial_transform: List[object] = []

    if crop_type == "random":
        spatial_transform.append(
            transforms.RandomResizedCrop(
                size,
                scale=(crop_min_scale, 1.0),
                ratio=(crop_min_ratio, 1.0 / crop_min_ratio),
            )
        )
        logger.info(
            f"Use RandomResizedCrop with the spatial size {size}, "
            f"scale {(crop_min_scale, 1.0)} "
            f"and ratio {(crop_min_ratio, 1.0 / crop_min_ratio)}."
        )
    elif crop_type == "center":
        spatial_transform.append(transforms.Resize(size))
        spatial_transform.append(transforms.CenterCrop(size))
        logger.info(f"Use Resize and CenterCrop with the spatial size {size}")
    elif crop_type == "corner":
        for _ in range(1, 5):
            scales.append(scales[-1] * scale_step)
        spatial_transform.append(transforms.MultiScaleCornerCrop(size, scales))
        logger.info(
            f"Use MultiScaleCornerCrop with the spatial size {size} "
            f"and scales {scales}"
        )
    else:
        message = "crop_type should be selected from ['random', 'center', 'corner']."
        logger.error(message)
        raise ValueError(message)

    if hflip:
        spatial_transform.append(transforms.RandomHorizontalFlip())
        logger.info("Use RandomHorizontalFlip")

    if vflip:
        spatial_transform.append(transforms.RandomVerticalFlip())
        logger.info("Use RandomVerticalFlip")

    if colorjitter:
        spatial_transform.append(
            transforms.ColorJitter(brightness, contrast, saturation, hue)
        )
        logger.info(
            f"Use ColorJitter (brightness = {brightness}, contrast = {contrast}, "
            f"saturation = {saturation} and hue = {hue})."
        )

    spatial_transform.append(transforms.ConvertImageDtype(torch.float32))
    spatial_transform.append(transforms.Normalize(mean=get_mean(), std=get_std()))

    return nn.Sequential(*spatial_transform).to(device)
