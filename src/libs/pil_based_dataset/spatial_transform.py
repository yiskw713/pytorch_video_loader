import random
from logging import getLogger
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import transforms

from ..mean_std import get_mean, get_std

logger = getLogger(__name__)


class Compose(transforms.Compose):
    def randomize_parameters(self) -> None:
        for t in self.transforms:
            t.randomize_parameters()


class ToTensor(transforms.ToTensor):
    def randomize_parameters(self) -> None:
        pass


class Normalize(transforms.Normalize):
    def randomize_parameters(self) -> None:
        pass


class ScaleValue(object):
    """
    If you set s to 255, the range of inputs is [0-255].
    """

    def __init__(self, s: int) -> None:
        self.s = s

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor *= self.s
        return tensor

    def randomize_parameters(self) -> None:
        pass


class Resize(transforms.Resize):
    # If size is an int, smaller edge of the image will be matched to this number.
    def randomize_parameters(self) -> None:
        pass


class Scale(transforms.Scale):
    def randomize_parameters(self) -> None:
        pass


class CenterCrop(transforms.CenterCrop):
    def randomize_parameters(self) -> None:
        pass


class CornerCrop(object):
    CROP_POSITIONS = ["c", "tl", "tr", "bl", "br"]

    def __init__(
        self,
        size: int,
        crop_position: Optional[str] = None,
    ):
        self.size = size
        self.crop_position = crop_position

        if crop_position is None:
            self.randomize = True
        else:
            self.randomize = False
        self.randomize_parameters()

    def __call__(self, img: Image.Image) -> Image.Image:
        image_width = img.size[0]
        image_height = img.size[1]

        h, w = (self.size, self.size)
        if self.crop_position == "c":
            i = int(round((image_height - h) / 2.0))
            j = int(round((image_width - w) / 2.0))
        elif self.crop_position == "tl":
            i = 0
            j = 0
        elif self.crop_position == "tr":
            i = 0
            j = image_width - self.size
        elif self.crop_position == "bl":
            i = image_height - self.size
            j = 0
        elif self.crop_position == "br":
            i = image_height - self.size
            j = image_width - self.size

        img = F.crop(img, i, j, h, w)

        return img

    def randomize_parameters(self) -> None:
        if self.randomize:
            self.crop_position = self.CROP_POSITIONS[
                random.randint(0, len(self.CROP_POSITIONS) - 1)
            ]

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + "(size={0}, crop_position={1}, randomize={2})".format(
                self.size, self.crop_position, self.randomize
            )
        )


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p)
        self.randomize_parameters()

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img: Image to be flipped.
        Returns:
            Randomly flipped image.
        """
        if self.random_p < self.p:
            return F.hflip(img)
        return img

    def randomize_parameters(self) -> None:
        self.random_p = random.random()


class RandomVerticalFlip(transforms.RandomVerticalFlip):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p)
        self.randomize_parameters()

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.random_p < self.p:
            return F.vflip(img)
        return img

    def randomize_parameters(self) -> None:
        self.random_p = random.random()


class MultiScaleCornerCrop(object):
    CROP_POSITIONS = ["c", "tl", "tr", "bl", "br"]

    def __init__(
        self,
        size: int,
        scales: List[float],
        interpolation: int = Image.BILINEAR,
    ):
        self.size = size
        self.scales = scales
        self.interpolation = interpolation

        self.randomize_parameters()

    def __call__(self, img: Image.Image) -> Image.Image:
        short_side = min(img.size[0], img.size[1])
        crop_size = int(short_side * self.scale)
        self.corner_crop.size = crop_size

        img = self.corner_crop(img)
        return img.resize((self.size, self.size), self.interpolation)

    def randomize_parameters(self) -> None:
        self.scale = self.scales[random.randint(0, len(self.scales) - 1)]
        crop_position = self.CROP_POSITIONS[
            random.randint(0, len(self.CROP_POSITIONS) - 1)
        ]

        self.corner_crop = CornerCrop(self.size, crop_position)

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + "(size={0}, scales={1}, interpolation={2})".format(
                self.size, self.scales, self.interpolation
            )
        )


class RandomResizedCrop(transforms.RandomResizedCrop):
    def __init__(
        self,
        size: int,
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        interpolation: int = Image.BILINEAR,
    ) -> None:
        super().__init__(size, scale, ratio, interpolation)
        self.randomize = True
        self.randomize_parameters()

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.randomize:
            self.random_crop = self.get_params(img, self.scale, self.ratio)
            self.randomize = False

        i, j, h, w = self.random_crop
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def randomize_parameters(self) -> None:
        self.randomize = True


class ColorJitter(transforms.ColorJitter):
    def __init__(
        self,
        brightness: float = 0,
        contrast: float = 0,
        saturation: float = 0,
        hue: float = 0,
    ) -> None:
        super().__init__(brightness, contrast, saturation, hue)
        self.randomize = True
        self.randomize_parameters()

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.randomize:
            self.transform = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue
            )
            self.randomize = False

        return self.transform(img)

    def randomize_parameters(self) -> None:
        self.randomize = True


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
    mean: List[float] = get_mean(),
    std: List[float] = get_std(),
) -> Compose:
    logger.info("Setting up spatial transform...")

    spatial_transform: List[object] = []
    if crop_type == "random":
        spatial_transform.append(
            RandomResizedCrop(
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
        spatial_transform.append(Resize(size))
        spatial_transform.append(CenterCrop(size))
        logger.info(f"Use Resize and CenterCrop with the spatial size {size}")
    elif crop_type == "corner":
        for _ in range(1, 5):
            scales.append(scales[-1] * scale_step)
        spatial_transform.append(MultiScaleCornerCrop(size, scales))
        logger.info(
            f"Use MultiScaleCornerCrop with the spatial size {size} "
            f"and scales {scales}"
        )
    else:
        message = "crop_type should be selected from ['random', 'center', 'corner']."
        logger.error(message)
        raise ValueError(message)

    if hflip:
        spatial_transform.append(RandomHorizontalFlip())
        logger.info("Use RandomHorizontalFlip")

    if vflip:
        spatial_transform.append(RandomVerticalFlip())
        logger.info("Use RandomVerticalFlip")

    if colorjitter:
        spatial_transform.append(ColorJitter(brightness, contrast, saturation, hue))
        logger.info(
            f"Use ColorJitter (brightness = {brightness}, contrast = {contrast}, "
            f"saturation = {saturation} and hue = {hue})."
        )

    spatial_transform.append(ToTensor())
    spatial_transform.append(Normalize(mean=mean, std=std))

    return Compose(spatial_transform)
