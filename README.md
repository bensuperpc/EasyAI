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

Launch training, save model and enable tensorboard:

```bash
python3 -m open_nsfw.py --save model.h5 --tensorboard
```

## New features

- Load and save model
- Data augmentation
- Argument parser

## Work in progress features

- Data set generator
- Better tensorboard integration

## Future features

- Lite version
- Docker image
- pip package
- Load and save weights

### Open source projects used

- [tensorflow](https://github.com/tensorflow/tensorflow)
- [tensorboard](https://github.com/tensorflow/tensorboard)
- [opencv](https://github.com/opencv/opencv)
- [git](https://github.com/git/git)
- [docker](https://github.com/docker/docker)
- [actions](https://github.com/actions/virtual-environments)

## Licensing

[MIT License](LICENSE)
