# Convolutional Neural Networks

Convolutional Neural Networks (CNN) project developed for Udacity's Deep Learning Nanodegree. The goal of this project is to build a pipeline that can be used within a web or mobile app to process real-world, user-supplied images. Given an image of a dog, the app will provide an estimate of the canine’s breed. If supplied an image of a human, the app will identify the resembling dog breed.

## Getting Started

### Setup Environment

#### Clone the Repository

``` batch
git clone https://github.com/geoglyph/dog-project.git
cd dog-project
```

#### Setup Linux

Tested on the following environment:

- Ubuntu 16.04.4 LTS
- NVIDIA GTX1080 (driver version 384.130)
- CUDA® Toolkit 9.0
- cuDNN v7.0

Create a Linux Conda environment with **CPU** backend and upgrade tensorflow:

``` batch
conda env create -f requirements/dog-linux.yml
conda activate dog-project
pip install --ignore-installed --upgrade tensorflow
KERAS_BACKEND=tensorflow python -c "from keras import backend"
python -m ipykernel install --user --name dog-project --display-name "dog-project"
 ```

Create a Linux Conda environment with **GPU** backend and upgrade tensorflow:

``` batch
conda env create -f requirements/dog-linux-gpu.yml
conda condaactivate dog-project
pip install --ignore-installed --upgrade tensorflow-gpu
KERAS_BACKEND=tensorflow python -c "from keras import backend"
python -m ipykernel install --user --name dog-project --display-name "dog-project"
```

#### Setup Windows

Tested on the following environment:

- Windows 10 Pro, 64-bit
- NVIDIA GTX1080 (driver version 385.54)
- CUDA® Toolkit 9.0
- cuDNN v7.0

Create a Windows Conda environment with **CPU** backend and upgrade tensorflow:

``` batch
conda env create -f requirements/dog-windows.yml
conda activate dog-project
pip install --ignore-installed --upgrade tensorflow
set KERAS_BACKEND=tensorflow
python -c "from keras import backend"
python -m ipykernel install --user --name dog-project --display-name "dog-project"
 ```

Create a Windows Conda environment with **GPU** backend and upgrade tensorflow:

``` batch
conda env create -f requirements/dog-windows-gpu.yml
conda activate dog-project
pip install --ignore-installed --upgrade tensorflow-gpu
set KERAS_BACKEND=tensorflow
python -c "from keras import backend"
python -m ipykernel install --user --name dog-project --display-name "dog-project"
```

#### Download Supporting Files
##### Datasets

Download the following datasets and copy to `dog-project/dogImages` and `dog-project/lfw` respectively.

[dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)

[human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip)

##### Pre-computed Bottleneck Features

Download the folloiing pre-computed bottleneck features and copy to `dog-project/bottleneck_features.

[VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz)

[Inception bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz)
