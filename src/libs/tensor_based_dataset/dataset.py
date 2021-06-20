import dataclasses
import glob
import io
import os
from logging import getLogger
from typing import Any, Dict, List, Optional

import h5py
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image

from ..temporal_transform import Compose as TemporalCompose

logger = getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class DatasetCSV:
    train: str
    val: str
    test: str


DATASET_CSV = {
    # paths from `src` directory
    "dummy": DatasetCSV(
        train="./csv/train.csv",
        val="./csv/val.csv",
        test="./csv/test.csv",
    ),
    "dummy2": DatasetCSV(
        train="./csv/train2.csv",
        val="./csv/val2.csv",
        test="./csv/test2.csv",
    ),
}


def get_dataloader(
    dataset_name: str,
    split: str,
    min_n_frames: int,
    video_format: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    device: str,
    drop_last: bool = False,
    spatial_transform: Optional[nn.Module] = None,
    temporal_transform: Optional[TemporalCompose] = None,
) -> DataLoader:

    if dataset_name not in DATASET_CSV:
        message = f"dataset_name should be selected from {list(DATASET_CSV.keys())}."
        logger.error(message)
        raise ValueError(message)

    if split not in ["train", "val", "test"]:
        message = "split should be selected from ['train', 'val', 'test']."
        logger.error(message)
        raise ValueError(message)

    csv_file = getattr(DATASET_CSV[dataset_name], split)

    data = VideoDataset(
        csv_file,
        min_n_frames,
        video_format,
        device=device,
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
    )
    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    logger.info(
        f"Dataset: {dataset_name}\tSplit: {split}"
        f"\tThe number of data: {len(data)}\tBatch size: {batch_size}."
    )

    return dataloader


class VideoLoader(object):
    """
    Return sequential frames in video clips corresponding to frame_indices.
    """

    def __init__(
        self,
        video_format="hdf5",
        temporal_transform: Optional[TemporalCompose] = None,
    ) -> None:
        super().__init__()
        self.temporal_transform = temporal_transform

        if video_format not in ("hdf5", "jpg", "png"):
            raise ValueError("Invalid video format.")

        self.video_format = video_format

    def __call__(self, video_path: str) -> torch.Tensor:
        """read videos
        Args:
            video_path
        Return:
            video: torch.Tensor (T, 3, H, W)
        """
        if self.video_format == "hdf5":
            video = self._read_video_from_hdf5(video_path)
        else:
            video = self._read_video_from_imgs(video_path)

        return video

    def _read_video_from_hdf5(self, video_path: str) -> torch.Tensor:
        with h5py.File(video_path, "r") as f:
            video_data = f["video"]
            frame_indices = [i for i in range(len(video_data))]
            if self.temporal_transform is not None:
                frame_indices = self.temporal_transform(frame_indices)

            video = []
            for i in frame_indices:
                img = Image.open(io.BytesIO(video_data[i]))
                tensor = transforms.functional.to_tensor(img)
                video.append(tensor)

        video_tensor = torch.stack(video, dim=0)
        return video_tensor

    def _read_video_from_imgs(self, video_dir_path: str) -> torch.Tensor:
        image_paths = glob.glob(os.path.join(video_dir_path, "*.jpg"))
        image_paths += glob.glob(os.path.join(video_dir_path, "*.png"))
        image_paths.sort()

        frame_indices = [i for i in range(len(image_paths))]
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        video = []
        for i in frame_indices:
            img = read_image(image_paths[i])
            video.append(img)

        video_tensor = torch.stack(video, dim=0)
        return video_tensor


class VideoDataset(Dataset):
    """Dataset class for Video Datset."""

    def __init__(
        self,
        csv_file: str,
        min_n_frames: int,
        video_format,
        device: str,
        spatial_transform: Optional[nn.Module] = None,
        temporal_transform: Optional[TemporalCompose] = None,
    ) -> None:
        super().__init__()
        logger.info("Setting up dataset.")
        logger.info(f"Loading {csv_file}...")

        self.df = pd.read_csv(csv_file)
        self.min_n_frames = min_n_frames
        self.device = device
        self.loader = VideoLoader(
            video_format=video_format, temporal_transform=temporal_transform
        )

        self.spatial_transform = spatial_transform
        if self.spatial_transform is not None:
            self.spatial_transform = self.spatial_transform.to(device)

        self._ignore_small_video()
        if len(self.df) == 0:
            message = "n_min_frames is too large. You must set a smaller value."
            logger.error(message)
            raise ValueError(message)

    def _ignore_small_video(self) -> None:
        self.df = self.df[self.df["n_frames"] >= self.min_n_frames]
        logger.info(
            f"Videos which have fewer than {self.min_n_frames} frames will never used."
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_path = self.df.iloc[idx]["video_path"]
        name = os.path.splitext(os.path.basename(video_path))[0]

        clip = self.loader(video_path)
        clip = clip.to(self.device)

        if self.spatial_transform is not None:
            clip = self.spatial_transform(clip)

        sample = {
            "clip": clip,
            "name": name,
        }

        return sample
