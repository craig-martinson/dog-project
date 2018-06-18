# Convolutional Neural Networks

Convolutional Neural Networks (CNN) project developed for Udacity's Deep Learning Nanodegree. The goal of this project is to build a pipeline that can be used within a web or mobile app to process real-world, user-supplied images. Given an image of a dog, the app will provide an estimate of the canine’s breed. If supplied an image of a human, the app will identify the resembling dog breed.

## Getting Started

### Environment

Tested on the following environment:

#### Windows

- Windows 10 Pro, 64-bit
- NVIDIA GTX1080 (driver version 385.54)
- CUDA® Toolkit 9.0
- cuDNN v7.0

#### Setup Environment

Create a conda environment with CPU backend and upgrade tensorflow:

``` batch
conda env create -f requirements/dog-windows.yml
activate dog-project
pip install --ignore-installed --upgrade tensorflow
 ```

Create a conda environment with GPU backend and upgrade tensorflow:

``` batch
conda env create -f requirements/dog-windows-gpu.yml
activate dog-project
pip install --ignore-installed --upgrade tensorflow-gpu
```

Clone the repository:

``` batch
git clone https://github.com/geoglyph/dog-project.git
cd dog-project
```

Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `dog-project/dogImages`

Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the repo, at location `dog-project/lfw`

Donwload the [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) for the dog dataset.  Place it in the repo, at location `dog-project/bottleneck_features`