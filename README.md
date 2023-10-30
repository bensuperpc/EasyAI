# EasyAI

## _Make your own AI easily ! via tensorflow_

## Description

The main goal of this project is to create an AI easily using tensorflow.

## Software requirements

- [Python 3.6+](https://www.python.org/downloads/)
- [Tensorflow 2.4+](https://www.tensorflow.org/install)
- [Tensorboard 2.4+](https://www.tensorflow.org/tensorboard/get_started)
- [OpenCV 4.5+](https://pypi.org/project/opencv-python/)
- [Git](https://git-scm.com/downloads)
- [Docker](https://docs.docker.com/get-docker/)

## Hardware requirements

We recommend using a GPU with Hardware Acceleration for Tensorflow.

| Hardware | minimum | recommended |
| --- | --- | --- |
| CPU | 2 cores | 4 cores |
| RAM | 4 GB | 8 GB |
| GPU | 2 GB | 4 GB |

## Usage

Get help:

```bash
python3 EasyAI.py --help
```

Launch training:

```bash
python3 EasyAI.py
```

Launch training without GPU:

```bash
python3 EasyAI.py --no-gpu
```

Launch training, save model and set dataset path:

```bash
python EasyAI.py --data_dir ./dataset/flower_photos --save test_AI.h5
```

## Example

Train a model with flowers dataset:

```bash
python EasyAI.py --save test_AI.h5
```

Predict images (**class_name order is important**):

```bash
python EasyAI.py --load test_AI.h5 --predict ./dataset/flower_photos/roses/ --class_name daisy dandelion roses sunflowers tulips
```

## Command table

| Command | Description | Default | Example |
| --- | --- | --- | --- |
| --data_dir | Path to the data directory. | ./dataset/ | --data_dir ./dataset/ |
| --save-model | Save model to a HDF5 file. | None | --save test_AI.h5 |
| --load-model | Load model from a HDF5 file. | None | --load test_AI.h5 |
| --predict | Predict images. | None | --predict ./dataset/flower_photos/roses/ |
| --class_name | Class name. | None | --class_name daisy dandelion roses sunflowers tulips |
| --no-gpu | Disable GPU. | False | --no-gpu |
| --batch_size | Batch size. | 32 | --batch_size 32 |
| --epochs | Number of epochs. | 10 | --epochs 10 |
| --model_path | Path to the model. | None | --model_path ./model/ |
| --tensorboard | Enable tensorboard (Slow). | False | --tensorboard |
| --checkpoint | Enable checkpoint. | False | --checkpoint |

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
