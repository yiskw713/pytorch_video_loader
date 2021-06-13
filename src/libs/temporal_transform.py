import random
from logging import getLogger
from typing import Any, List

logger = getLogger(__name__)


class Compose(object):
    def __init__(self, transforms: List[Any]) -> None:
        self.transforms = transforms

    def __call__(self, frame_indices: List[int]) -> List[int]:
        for transform in self.transforms:
            frame_indices = transform(frame_indices)
        return frame_indices


class LoopPadding(object):
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, frame_indices: List[int]) -> List[int]:
        for index in frame_indices:
            if len(frame_indices) >= self.size:
                break
            frame_indices.append(index)

        return frame_indices


class ReverseLoopPadding(object):
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, frame_indices: List[int]) -> List[int]:
        """Reverse Loop Padding
        e.g.) frame_indices = [4, 5, 6, 7] and size = 8
        [4, 5, 6, 7]
        -> [4, 5, 6, 7, 6, 5, 4]
        -> [4, 5, 6, 7, 6, 5, 4, 5, 6, 7, 6, 5, 4]
        -> [4, 5, 6, 7, 6, 5, 4, 5]
        """
        while len(frame_indices) < self.size:
            frame_indices += list(reversed(frame_indices[1:]))

        return frame_indices[: self.size]


class TemporalBeginCrop(object):
    def __init__(self, size: int) -> None:
        self.size = size
        self.padding = ReverseLoopPadding(size)

    def __call__(self, frame_indices: List[int]) -> List[int]:
        out = frame_indices[: self.size]

        if len(out) < self.size:
            out = self.padding(out)

        return out


class TemporalCenterCrop(object):
    def __init__(self, size: int) -> None:
        self.size = size
        self.padding = ReverseLoopPadding(size)

    def __call__(self, frame_indices: List[int]) -> List[int]:
        center_index = len(frame_indices) // 2
        begin_index = max(0, center_index - (self.size // 2))
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        if len(out) < self.size:
            out = self.padding(out)

        return out


class TemporalRandomCrop(object):
    def __init__(self, size):
        self.size = size
        self.padding = ReverseLoopPadding(size)

    def __call__(self, frame_indices: List[int]) -> List[int]:
        rand_end = max(0, len(frame_indices) - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        if len(out) < self.size:
            out = self.padding(out)

        return out


class TemporalSubsampling(object):
    def __init__(self, stride: int) -> None:
        self.stride = stride

    def __call__(self, frame_indices: List[int]) -> List[int]:
        return frame_indices[:: self.stride]


def get_temporal_transform(
    temp_crop_type: str,
    n_frames: int,
    temp_downsamp_rate: int = 1,
) -> Compose:
    logger.info("Setting up temporal transform...")

    temporal_transform: List[Any] = []
    if temp_downsamp_rate > 1:
        logger.info(
            f"Use TemporalSubsampling with temp_downsamp_rate {temp_downsamp_rate}."
        )
        temporal_transform.append(TemporalSubsampling(n_frames))

    if temp_crop_type == "random":
        logger.info(f"Use TemporalRandomCrop with n_frames {n_frames}")
        temporal_transform.append(TemporalRandomCrop(n_frames))
    elif temp_crop_type == "center":
        logger.info(f"Use TemporalCenterCrop with n_frames {n_frames}")
        temporal_transform.append(TemporalCenterCrop(n_frames))
    else:
        message = "temp_crop_type should be selected from ['random', 'center']."
        logger.error(message)
        raise ValueError(message)

    return Compose(temporal_transform)
