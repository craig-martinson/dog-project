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
pip install flask gevent requests pillow
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.8.0-cp36-cp36m-linux_x86_64.whl
KERAS_BACKEND=tensorflow python -c "from keras import backend"
python -m ipykernel install --user --name dog-project --display-name "dog-project"
 ```

Create a Linux Conda environment with **GPU** backend and upgrade tensorflow:

``` batch
conda env create -f requirements/dog-linux-gpu.yml
conda activate dog-project
pip install flask gevent requests pillow
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.8.0-cp36-cp36m-linux_x86_64.whl
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
pip install flask gevent requests pillow
pip install --ignore-installed --upgrade tensorflow
set KERAS_BACKEND=tensorflow
python -c "from keras import backend"
python -m ipykernel install --user --name dog-project --display-name "dog-project"
 ```

Create a Windows Conda environment with **GPU** backend and upgrade tensorflow:

``` batch
conda env create -f requirements/dog-windows-gpu.yml
conda activate dog-project
pip install flask gevent requests pillow
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

Download the following pre-computed bottleneck features and copy to `dog-project/bottleneck_features.

[VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz)

[Inception bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz)

## Jupyter Notebooks

The following jupyter notebooks were developed to support this project:

Description | Link
--- | ---
Project notebook provided by Udacity, demonstrates transfer learning with Keras | [Dog App Notebook](./dog_app/dog_app.md)
Demonstrates the use of data augmentation with transfer learning with Keras | [Dog App Augmented Notebook](./dog_app_augmented/dog_app_augmented.md)

## Python Programs

The following python programs were developed to support this project:

Description | File
--- | ---
Runs inference on each file in /images folder using saved InceptionV3 model | [predict_breed.py](./predict_breed.py)
Uses [flask](http://flask.pocoo.org/) to host simple REST API providing inference on saved InceptionV3 model | [predict_breed_server.py](./predict_breed.py)
Used to test REST API, sends each file in /images folder to REST API for inference | [server_api_test.py](./server_api_testpy)

## References

The following resources were used in developing this project:

Usage | Link
--- | ---
Python code used to visualise loss history when training a Keras model | [Visualize Loss History](https://chrisalbon.com/deep_learning/keras/visualize_loss_history/)
Keras data augmentation example<br>refer: cifar10-augmentation/cifar10_augmentation.ipynb<br>Used in development of data augmentaiton code in Jupyter notebooks  | [AIND Term 2 -- Lesson on Convolutional Neural Networks](https://github.com/udacity/aind2-cnn)
Keras bottleneck feature extraction<br>refer: transfer-learning/bottleneck_features.ipynb<br>Used in development of feature extraction code in Jupyter notebooks | [AIND Term 2 -- Lesson on Convolutional Neural Networks](https://github.com/udacity/aind2-cnn)
Keras bottleneck feature extraction with data augmentation<br>Used as initial inspiration for [Dog App Augmented](./dog_app_augmented/dog_app_augmented.md) | [The Keras Blog](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
Deploying a Keras model as a REST API<br>Used as initial inspiration for [predict_breed_server.py](./predict_breed.py) | [The Keras Blog](https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html)
