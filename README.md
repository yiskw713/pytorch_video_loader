# Video Loader with PyTorch

Implementation of two types of video loaders with pytorch;

* PIL-based video loader (pre-processing videos as PIL images)
* Tensor-based video loader (pre-processing videos as torch.Tensor)

## Dependencies

* python 3.x
* pytorch
* torchvision
* h5py

You can install the necessary packages by running `poetry install`.

## Comparing two video loaders

```sh
$ cd src
$ poetry run python main.py
```

## Results

### both on CPU

```console
Device: cpu     n_samples: 16   n_frames: 16.
PIL-based dataloader: Ave.  11.66 sec.
tensor-based dataloader: Ave.  12.22 sec.
```

### PIL-based video loader on CPU vs Tensor-based video loader on GPU

```console
Device: cuda	n_samples: 16	n_frames: 16.
PIL-based dataloader: Ave.  1.63 sec.
tensor-based dataloader: Ave.  4.62 sec.
```

## TODO

- [ ] test code
- [ ] comparison using JPG Video
