import time

import torch

from libs.device import get_device
from libs.pil_based_dataset.dataset import get_dataloader as get_pil_dataloader
from libs.pil_based_dataset.spatial_transform import (
    get_spatial_transform as get_pil_spatial_transform,
)
from libs.temporal_transform import get_temporal_transform
from libs.tensor_based_dataset.dataset import get_dataloader as get_tensor_dataloader
from libs.tensor_based_dataset.spatial_transform import (
    get_spatial_transform as get_tensor_spatial_transform,
)


def main() -> None:
    # to avoid the error 'Cannot re-initialize CUDA in forked subprocess'
    # https://github.com/pytorch/pytorch/issues/40403
    torch.multiprocessing.set_start_method("spawn")

    device = get_device(allow_only_gpu=False)

    temporal_transform = get_temporal_transform(
        temp_crop_type="random",
        n_frames=16,
        temp_downsamp_rate=1,
    )

    pil_spatial_transform = get_pil_spatial_transform(
        crop_type="random",
        size=224,
        hflip=True,
        vflip=False,
        colorjitter=False,
    )

    tensor_spatial_transform = get_tensor_spatial_transform(
        crop_type="random",
        size=224,
        hflip=True,
        vflip=False,
        colorjitter=False,
        device=device,
    )

    pil_loader = get_pil_dataloader(
        dataset_name="dummy",
        split="train",
        min_n_frames=64,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
        drop_last=False,
        spatial_transform=pil_spatial_transform,
        temporal_transform=temporal_transform,
    )

    tensor_loader = get_tensor_dataloader(
        dataset_name="dummy",
        split="train",
        min_n_frames=64,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
        device=device,
        drop_last=False,
        spatial_transform=tensor_spatial_transform,
        temporal_transform=temporal_transform,
    )

    # measure loading time
    print("-" * 10, "Start loading data", "-" * 10)
    n_epochs = 5
    pil_time = 0.0
    tensor_time = 0.0

    # PIL-based dataset
    for _ in range(n_epochs):
        start = time.time()

        for sample in pil_loader:
            pass

        pil_time += time.time() - start

    # tensor-based dataset
    for _ in range(n_epochs):
        start = time.time()

        for sample in tensor_loader:
            pass

        tensor_time += time.time() - start

    pil_time /= n_epochs
    tensor_time /= n_epochs

    print(f"Device: {device}\tn_samples: {len(pil_loader.dataset)}\tn_frames: {16}.")
    print(f"PIL-based dataloader: Ave. {pil_time: .2f} sec.")
    print(f"tensor-based dataloader: Ave. {tensor_time: .2f} sec.")


if __name__ == "__main__":
    main()
