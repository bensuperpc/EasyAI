# open_nsfw

## _Not Suitable for Work (NSFW) image detector via tensorflow_

[![open_nsfw](https://github.com/bensuperpc/open_nsfw/actions/workflows/base.yml/badge.svg)](https://github.com/bensuperpc/open_nsfw/actions/workflows/base.yml)

## Usage

Get help:

```bash
python3 -m open_nsfw.py --help
```

Launch training:

```bash
python3 -m open_nsfw.py
```

Launch training without GPU:

```bash
python3 -m open_nsfw.py --no-gpu
```

Launch training, save model and set dataset path:

```bash
python open_nsfw.py --data_dir ./dataset/flower_photos --save test_AI.h5
```

## Example

Train a model with flowers dataset:

```bash
python open_nsfw.py --save test_AI.h5
```

Predict images (class_name order is important):

```bash
python open_nsfw.py --load test_AI.h5 --predict ./dataset/flower_photos/roses/ --class_name daisy dandelion roses sunflowers tulips
```

## Done features

- Working model
- Tensorboard integration
- GPU support
- Load and save model
- Data augmentation
- Argument parser

## Work in progress features

- Data set generator
- Load and save weights

## Future features

- Lite version
- Docker image
- pip package


### Open source projects used

- [tensorflow](https://github.com/tensorflow/tensorflow)
- [tensorboard](https://github.com/tensorflow/tensorboard)
- [opencv](https://github.com/opencv/opencv)
- [git](https://github.com/git/git)
- [docker](https://github.com/docker/docker)
- [actions](https://github.com/actions/virtual-environments)

## Licensing

[MIT License](LICENSE)
