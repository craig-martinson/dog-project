
# Artificial Intelligence Nanodegree

## Convolutional Neural Networks

## Project: Write an Algorithm for a Dog Identification App 

---

In this notebook, some template code has already been provided for you, and you will need to implement additional functionality to successfully complete this project. You will not need to modify the included code beyond what is requested. Sections that begin with **'(IMPLEMENTATION)'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section, and the specifics of the implementation are marked in the code block with a 'TODO' statement. Please be sure to read the instructions carefully! 

> **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.

>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut.  Markdown cells can be edited by double-clicking the cell to enter edit mode.

The rubric contains _optional_ "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. If you decide to pursue the "Stand Out Suggestions", you should include the code in this IPython notebook.



---
### Why We're Here 

In this notebook, you will make the first steps towards developing an algorithm that could be used as part of a mobile or web app.  At the end of this project, your code will accept any user-supplied image as input.  If a dog is detected in the image, it will provide an estimate of the dog's breed.  If a human is detected, it will provide an estimate of the dog breed that is most resembling.  The image below displays potential sample output of your finished project (... but we expect that each student's algorithm will behave differently!). 

![Sample Dog Output](images/sample_dog_output.png)

In this real-world setting, you will need to piece together a series of models to perform different tasks; for instance, the algorithm that detects humans in an image will be different from the CNN that infers dog breed.  There are many points of possible failure, and no perfect algorithm exists.  Your imperfect solution will nonetheless create a fun user experience!

### The Road Ahead

We break the notebook into separate steps.  Feel free to use the links below to navigate the notebook.

* [Step 0](#step0): Import Datasets
* [Step 1](#step1): Detect Humans
* [Step 2](#step2): Detect Dogs
* [Step 3](#step3): Create a CNN to Classify Dog Breeds (from Scratch)
* [Step 4](#step4): Use a CNN to Classify Dog Breeds (using Transfer Learning)
* [Step 5](#step5): Create a CNN to Classify Dog Breeds (using Transfer Learning)
* [Step 6](#step6): Write your Algorithm
* [Step 7](#step7): Test Your Algorithm

---
<a id='step0'></a>
## Step 0: Import Datasets

### Import Dog Dataset

In the code cell below, we import a dataset of dog images.  We populate a few variables through the use of the `load_files` function from the scikit-learn library:
- `train_files`, `valid_files`, `test_files` - numpy arrays containing file paths to images
- `train_targets`, `valid_targets`, `test_targets` - numpy arrays containing onehot-encoded classification labels 
- `dog_names` - list of string-valued dog breed names for translating labels


```python
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
import utility

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))

```

    There are 133 total dog categories.
    There are 8351 total dog images.
    
    There are 6680 training dog images.
    There are 835 validation dog images.
    There are 836 test dog images.



```python
# check version of tensorflow and keras
import tensorflow as tf
import keras as keras

print("Tensorflow version: {}".format(tf.__version__))
print("Keras version: {}".format(keras.__version__))

# ignore tensorflow warnings
tf.logging.set_verbosity(tf.logging.ERROR)
```

    Tensorflow version: 1.8.0
    Keras version: 2.0.2


### Import Human Dataset

In the code cell below, we import a dataset of human images, where the file paths are stored in the numpy array `human_files`.


```python
import random
random.seed(8675309)

# load filenames in shuffled human dataset
human_files = np.array(glob("lfw/*/*"))
random.shuffle(human_files)

# print statistics about the dataset
print('There are %d total human images.' % len(human_files))
```

    There are 13233 total human images.



```python
print(human_files[3])
```

    lfw/Robert_Mueller/Robert_Mueller_0002.jpg


---
<a id='step1'></a>
## Step 1: Detect Humans

We use OpenCV's implementation of [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) to detect human faces in images.  OpenCV provides many pre-trained face detectors, stored as XML files on [github](https://github.com/opencv/opencv/tree/master/data/haarcascades).  We have downloaded one of these detectors and stored it in the `haarcascades` directory.

In the next code cell, we demonstrate how to use this detector to find human faces in a sample image.


```python
import cv2                
import matplotlib.pyplot as plt                        
%matplotlib inline                               

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[3])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()
```

    Number of faces detected: 1



![png](output_7_1.png)


Before using any of the face detectors, it is standard procedure to convert the images to grayscale.  The `detectMultiScale` function executes the classifier stored in `face_cascade` and takes the grayscale image as a parameter.  

In the above code, `faces` is a numpy array of detected faces, where each row corresponds to a detected face.  Each detected face is a 1D array with four entries that specifies the bounding box of the detected face.  The first two entries in the array (extracted in the above code as `x` and `y`) specify the horizontal and vertical positions of the top left corner of the bounding box.  The last two entries in the array (extracted here as `w` and `h`) specify the width and height of the box.

### Write a Human Face Detector

We can use this procedure to write a function that returns `True` if a human face is detected in an image and `False` otherwise.  This function, aptly named `face_detector`, takes a string-valued file path to an image as input and appears in the code block below.


```python
# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0
```

### (IMPLEMENTATION) Assess the Human Face Detector

__Question 1:__ Use the code cell below to test the performance of the `face_detector` function.  
- What percentage of the first 100 images in `human_files` have a detected human face?  
- What percentage of the first 100 images in `dog_files` have a detected human face? 

Ideally, we would like 100% of human images with a detected face and 0% of dog images with a detected face.  You will see that our algorithm falls short of this goal, but still gives acceptable performance.  We extract the file paths for the first 100 images from each of the datasets and store them in the numpy arrays `human_files_short` and `dog_files_short`.

__Answer:__ 


```python
human_files_short = human_files[:100]
dog_files_short = train_files[:100]
# Do NOT modify the code above this line.

## Test the performance of the face_detector algorithm 
## on the images in human_files_short and dog_files_short.

human_count = 0

for img in human_files_short:
    human_count += 1 if face_detector(img) else 0
    
print("Percentage of human faces detected in human_files_short: {}%".format(100.0 * human_count / len(human_files_short)))

human_count = 0

for img in dog_files_short:
    human_count += 1 if face_detector(img) else 0
    
print("Percentage of human faces detected in dog_files_short: {}%".format(100.0 * human_count / len(dog_files_short)))
```

    Percentage of human faces detected in human_files_short: 99.0%
    Percentage of human faces detected in dog_files_short: 11.0%


__Question 2:__ This algorithmic choice necessitates that we communicate to the user that we accept human images only when they provide a clear view of a face (otherwise, we risk having unneccessarily frustrated users!). In your opinion, is this a reasonable expectation to pose on the user? If not, can you think of a way to detect humans in images that does not necessitate an image with a clearly presented face?

__Answer:__
As usual it depends on the application of the technology. There maybe scenarios where a clear view of the face is simply not possible or convenient. In these scenarios the algorithm could utilise other visible features such as shape to determine if the image is of a human or not.

We suggest the face detector from OpenCV as a potential way to detect human images in your algorithm, but you are free to explore other approaches, especially approaches that make use of deep learning :).  Please use the code cell below to design and test your own face detection algorithm.  If you decide to pursue this _optional_ task, report performance on each of the datasets.


```python
## (Optional) TODO: Report the performance of another  
## face detection algorithm on the LFW dataset
### Feel free to use as many code cells as needed.
```

---
<a id='step2'></a>
## Step 2: Detect Dogs

In this section, we use a pre-trained [ResNet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006) model to detect dogs in images.  Our first line of code downloads the ResNet-50 model, along with weights that have been trained on [ImageNet](http://www.image-net.org/), a very large, very popular dataset used for image classification and other vision tasks.  ImageNet contains over 10 million URLs, each linking to an image containing an object from one of [1000 categories](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).  Given an image, this pre-trained ResNet-50 model returns a prediction (derived from the available categories in ImageNet) for the object that is contained in the image.


```python
from keras.applications.resnet50 import ResNet50

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')
```

### Pre-process the Data

When using TensorFlow as backend, Keras CNNs require a 4D array (which we'll also refer to as a 4D tensor) as input, with shape

$$
(\text{nb_samples}, \text{rows}, \text{columns}, \text{channels}),
$$

where `nb_samples` corresponds to the total number of images (or samples), and `rows`, `columns`, and `channels` correspond to the number of rows, columns, and channels for each image, respectively.  

The `path_to_tensor` function below takes a string-valued file path to a color image as input and returns a 4D tensor suitable for supplying to a Keras CNN.  The function first loads the image and resizes it to a square image that is $224 \times 224$ pixels.  Next, the image is converted to an array, which is then resized to a 4D tensor.  In this case, since we are working with color images, each image has three channels.  Likewise, since we are processing a single image (or sample), the returned tensor will always have shape

$$
(1, 224, 224, 3).
$$

The `paths_to_tensor` function takes a numpy array of string-valued image paths as input and returns a 4D tensor with shape 

$$
(\text{nb_samples}, 224, 224, 3).
$$

Here, `nb_samples` is the number of samples, or number of images, in the supplied array of image paths.  It is best to think of `nb_samples` as the number of 3D tensors (where each 3D tensor corresponds to a different image) in your dataset!


```python
from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
```

### Making Predictions with ResNet-50

Getting the 4D tensor ready for ResNet-50, and for any other pre-trained model in Keras, requires some additional processing.  First, the RGB image is converted to BGR by reordering the channels.  All pre-trained models have the additional normalization step that the mean pixel (expressed in RGB as $[103.939, 116.779, 123.68]$ and calculated from all pixels in all images in ImageNet) must be subtracted from every pixel in each image.  This is implemented in the imported function `preprocess_input`.  If you're curious, you can check the code for `preprocess_input` [here](https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py).

Now that we have a way to format our image for supplying to ResNet-50, we are now ready to use the model to extract the predictions.  This is accomplished with the `predict` method, which returns an array whose $i$-th entry is the model's predicted probability that the image belongs to the $i$-th ImageNet category.  This is implemented in the `ResNet50_predict_labels` function below.

By taking the argmax of the predicted probability vector, we obtain an integer corresponding to the model's predicted object class, which we can identify with an object category through the use of this [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a). 


```python
from keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))
```

### Write a Dog Detector

While looking at the [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a), you will notice that the categories corresponding to dogs appear in an uninterrupted sequence and correspond to dictionary keys 151-268, inclusive, to include all categories from `'Chihuahua'` to `'Mexican hairless'`.  Thus, in order to check to see if an image is predicted to contain a dog by the pre-trained ResNet-50 model, we need only check if the `ResNet50_predict_labels` function above returns a value between 151 and 268 (inclusive).

We use these ideas to complete the `dog_detector` function below, which returns `True` if a dog is detected in an image (and `False` if not).


```python
### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 
```

### (IMPLEMENTATION) Assess the Dog Detector

__Question 3:__ Use the code cell below to test the performance of your `dog_detector` function.  
- What percentage of the images in `human_files_short` have a detected dog?  
- What percentage of the images in `dog_files_short` have a detected dog?

__Answer:__ 


```python
### Test the performance of the dog_detector function
### on the images in human_files_short and dog_files_short.

dog_count = 0

for img in human_files_short:
    dog_count += 1 if dog_detector(img) else 0
    
print("Percentage of dogs detected in human_files_short: {}%".format(100.0 * dog_count / len(human_files_short)))

dog_count = 0

for img in dog_files_short:
    dog_count += 1 if dog_detector(img) else 0
    
print("Percentage of dogs detected in dog_files_short: {}%".format(100.0 * dog_count / len(dog_files_short)))
```

    Percentage of dogs detected in human_files_short: 4.0%
    Percentage of dogs detected in dog_files_short: 100.0%


---
<a id='step3'></a>
## Step 3: Create a CNN to Classify Dog Breeds (from Scratch)

Now that we have functions for detecting humans and dogs in images, we need a way to predict breed from images.  In this step, you will create a CNN that classifies dog breeds.  You must create your CNN _from scratch_ (so, you can't use transfer learning _yet_!), and you must attain a test accuracy of at least 1%.  In Step 5 of this notebook, you will have the opportunity to use transfer learning to create a CNN that attains greatly improved accuracy.

Be careful with adding too many trainable layers!  More parameters means longer training, which means you are more likely to need a GPU to accelerate the training process.  Thankfully, Keras provides a handy estimate of the time that each epoch is likely to take; you can extrapolate this estimate to figure out how long it will take for your algorithm to train. 

We mention that the task of assigning breed to dogs from images is considered exceptionally challenging.  To see why, consider that *even a human* would have great difficulty in distinguishing between a Brittany and a Welsh Springer Spaniel.  

Brittany | Welsh Springer Spaniel
- | - 
<img src="images/Brittany_02625.jpg" width="100"> | <img src="images/Welsh_springer_spaniel_08203.jpg" width="200">

It is not difficult to find other dog breed pairs with minimal inter-class variation (for instance, Curly-Coated Retrievers and American Water Spaniels).  

Curly-Coated Retriever | American Water Spaniel
- | -
<img src="images/Curly-coated_retriever_03896.jpg" width="200"> | <img src="images/American_water_spaniel_00648.jpg" width="200">


Likewise, recall that labradors come in yellow, chocolate, and black.  Your vision-based algorithm will have to conquer this high intra-class variation to determine how to classify all of these different shades as the same breed.  

Yellow Labrador | Chocolate Labrador | Black Labrador
- | -
<img src="images/Labrador_retriever_06457.jpg" width="150"> | <img src="images/Labrador_retriever_06455.jpg" width="240"> | <img src="images/Labrador_retriever_06449.jpg" width="220">

We also mention that random chance presents an exceptionally low bar: setting aside the fact that the classes are slightly imabalanced, a random guess will provide a correct answer roughly 1 in 133 times, which corresponds to an accuracy of less than 1%.  

Remember that the practice is far ahead of the theory in deep learning.  Experiment with many different architectures, and trust your intuition.  And, of course, have fun! 

### Pre-process the Data

We rescale the images by dividing every pixel in every image by 255.


```python
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255
```

    100%|██████████| 6680/6680 [00:37<00:00, 178.27it/s]
    100%|██████████| 835/835 [00:04<00:00, 197.61it/s]
    100%|██████████| 836/836 [00:04<00:00, 200.46it/s]


### (IMPLEMENTATION) Model Architecture

Create a CNN to classify dog breed.  At the end of your code cell block, summarize the layers of your model by executing the line:
    
        model.summary()

We have imported some Python modules to get you started, but feel free to import as many modules as you need.  If you end up getting stuck, here's a hint that specifies a model that trains relatively fast on CPU and attains >1% test accuracy in 5 epochs:

![Sample CNN](images/sample_cnn.png)
           
__Question 4:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  If you chose to use the hinted architecture above, describe why you think that CNN architecture should work well for the image classification task.

__Answer:__ 

This CNN architecture is based on a stack of 6 convolution layers with increasing filter size. Each convolution layer has a Relu activation function in order to increase the non-linearity of the model. The purpose of the convolution layer is to transform the model from wide to deep.

To maintain a resonable training time, max pooling layers are used with each convolution layer to reduce complexity and reduce parameter count. The max pooling layer also improves the models ability to handle dogs that are not centred in the image (translation invariance). 

Following the max pooling layer, a batch normalization layer is used to help the model optimise, helping to reduce the number of required epochs. The batch normalization layer also introduces noise into the network assisting in the prevention of overfitting.

Note: To further mitigate overfitting, a small amount of drop out was introduced after each stack, however this proved detremental to overall training performance.

Following the convolution layers, a global average pooling layer is used to reduce the number of parameters prior to feeding into the final fully connected layer.

The final fully connected layer has a single node for each target class in the model and uses a softmax activation to represent the probability distribution across target classes (dog breeds).

The number of stacks, filter size and amount of drop-out was determined through trial and error. Initial problems with overfitting were addressed by augmenting the data, effectively increasing the amount of data available to train the network.

Various optimizers were trialed, with the most promising results from the adam optimizer.

This model has been trained up to 42% accuracy using 30 epochs on the test dataset. To increase accuracy significantly past this point requires further work to mitigate overfitting. This may require additional stacks, the use of different filter sizes or a combination of both.


```python
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense, BatchNormalization
from keras.models import Sequential
from keras.backend import eval

output_size = len(train_targets[0])

model = Sequential()

# stack 1
model.add(Conv2D(filters=16, kernel_size=3, input_shape=train_tensors[0].shape, activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(BatchNormalization())

# stack 2
model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(BatchNormalization())

# stack 3
model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(BatchNormalization())

# stack 4
model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(BatchNormalization())

# stack 5
model.add(Conv2D(filters=196, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(BatchNormalization())

# stack 6
model.add(Conv2D(filters=256, kernel_size=2, strides=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(BatchNormalization())

# final pooling layer
model.add(GlobalAveragePooling2D())

# final fully connected layer
model.add(Dense(output_size, activation='softmax'))

model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_7 (Conv2D)            (None, 222, 222, 16)      448       
    _________________________________________________________________
    max_pooling2d_9 (MaxPooling2 (None, 111, 111, 16)      0         
    _________________________________________________________________
    batch_normalization_7 (Batch (None, 111, 111, 16)      64        
    _________________________________________________________________
    conv2d_8 (Conv2D)            (None, 109, 109, 32)      4640      
    _________________________________________________________________
    max_pooling2d_10 (MaxPooling (None, 54, 54, 32)        0         
    _________________________________________________________________
    batch_normalization_8 (Batch (None, 54, 54, 32)        128       
    _________________________________________________________________
    conv2d_9 (Conv2D)            (None, 52, 52, 64)        18496     
    _________________________________________________________________
    max_pooling2d_11 (MaxPooling (None, 26, 26, 64)        0         
    _________________________________________________________________
    batch_normalization_9 (Batch (None, 26, 26, 64)        256       
    _________________________________________________________________
    conv2d_10 (Conv2D)           (None, 25, 25, 128)       32896     
    _________________________________________________________________
    max_pooling2d_12 (MaxPooling (None, 12, 12, 128)       0         
    _________________________________________________________________
    batch_normalization_10 (Batc (None, 12, 12, 128)       512       
    _________________________________________________________________
    conv2d_11 (Conv2D)           (None, 11, 11, 196)       100548    
    _________________________________________________________________
    max_pooling2d_13 (MaxPooling (None, 5, 5, 196)         0         
    _________________________________________________________________
    batch_normalization_11 (Batc (None, 5, 5, 196)         784       
    _________________________________________________________________
    conv2d_12 (Conv2D)           (None, 2, 2, 256)         200960    
    _________________________________________________________________
    max_pooling2d_14 (MaxPooling (None, 1, 1, 256)         0         
    _________________________________________________________________
    batch_normalization_12 (Batc (None, 1, 1, 256)         1024      
    _________________________________________________________________
    global_average_pooling2d_2 ( (None, 256)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 133)               34181     
    =================================================================
    Total params: 394,937.0
    Trainable params: 393,553.0
    Non-trainable params: 1,384.0
    _________________________________________________________________


### Compile the Model


```python
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### (IMPLEMENTATION) Train the Model

Train your model in the code cell below.  Use model checkpointing to save the model that attains the best validation loss.

You are welcome to [augment the training data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html), but this is not a requirement. 


```python
# refer https://github.com/udacity/aind2-cnn for details

from keras.preprocessing.image import ImageDataGenerator

# create and configure augmented image generator
datagen_train = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# fit augmented image generator on data
datagen_train.fit(train_tensors)
```


```python
utility.visualize_augmented_images(train_tensors, datagen_train, 12)
```


![png](output_33_0.png)



![png](output_33_1.png)



```python
from keras.callbacks import ModelCheckpoint  

### TODO: specify the number of epochs that you would like to use to train the model.

epochs = 30

### Do NOT modify the code below this line.

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)

#history = model.fit(train_tensors, train_targets, 
#          validation_data=(valid_tensors, valid_targets),
#          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)

# Code updated to use augmented dataset
batch_size = 20

history = model.fit_generator(datagen_train.flow(train_tensors, train_targets, batch_size=batch_size),
                    steps_per_epoch=train_tensors.shape[0] // batch_size,
                    epochs=epochs, verbose=1, callbacks=[checkpointer],
                    validation_data=(valid_tensors, valid_targets),
                    validation_steps=valid_tensors.shape[0] // batch_size)
```

    Epoch 1/30
    333/334 [============================>.] - ETA: 0s - loss: 4.8373 - acc: 0.0341Epoch 00000: val_loss improved from inf to 5.11709, saving model to saved_models/weights.best.from_scratch.hdf5
    334/334 [==============================] - 40s - loss: 4.8362 - acc: 0.0344 - val_loss: 5.1171 - val_acc: 0.0192
    Epoch 2/30
    333/334 [============================>.] - ETA: 0s - loss: 4.3979 - acc: 0.0547Epoch 00001: val_loss improved from 5.11709 to 4.96333, saving model to saved_models/weights.best.from_scratch.hdf5
    334/334 [==============================] - 38s - loss: 4.3992 - acc: 0.0545 - val_loss: 4.9633 - val_acc: 0.0551
    Epoch 3/30
    333/334 [============================>.] - ETA: 0s - loss: 4.1904 - acc: 0.0740Epoch 00002: val_loss improved from 4.96333 to 4.57271, saving model to saved_models/weights.best.from_scratch.hdf5
    334/334 [==============================] - 38s - loss: 4.1896 - acc: 0.0740 - val_loss: 4.5727 - val_acc: 0.0695
    Epoch 4/30
    333/334 [============================>.] - ETA: 0s - loss: 4.0036 - acc: 0.0964Epoch 00003: val_loss improved from 4.57271 to 4.17029, saving model to saved_models/weights.best.from_scratch.hdf5
    334/334 [==============================] - 40s - loss: 4.0041 - acc: 0.0961 - val_loss: 4.1703 - val_acc: 0.0922
    Epoch 5/30
    333/334 [============================>.] - ETA: 0s - loss: 3.8495 - acc: 0.1158Epoch 00004: val_loss did not improve
    334/334 [==============================] - 40s - loss: 3.8501 - acc: 0.1159 - val_loss: 4.4373 - val_acc: 0.0850
    Epoch 6/30
    333/334 [============================>.] - ETA: 0s - loss: 3.7128 - acc: 0.1332Epoch 00005: val_loss improved from 4.17029 to 3.80446, saving model to saved_models/weights.best.from_scratch.hdf5
    334/334 [==============================] - 40s - loss: 3.7116 - acc: 0.1334 - val_loss: 3.8045 - val_acc: 0.1257
    Epoch 7/30
    333/334 [============================>.] - ETA: 0s - loss: 3.5782 - acc: 0.1515Epoch 00006: val_loss improved from 3.80446 to 3.71981, saving model to saved_models/weights.best.from_scratch.hdf5
    334/334 [==============================] - 39s - loss: 3.5775 - acc: 0.1515 - val_loss: 3.7198 - val_acc: 0.1533
    Epoch 8/30
    333/334 [============================>.] - ETA: 0s - loss: 3.4922 - acc: 0.1664Epoch 00007: val_loss did not improve
    334/334 [==============================] - 40s - loss: 3.4942 - acc: 0.1662 - val_loss: 3.7497 - val_acc: 0.1533
    Epoch 9/30
    333/334 [============================>.] - ETA: 0s - loss: 3.3739 - acc: 0.1859Epoch 00008: val_loss improved from 3.71981 to 3.42739, saving model to saved_models/weights.best.from_scratch.hdf5
    334/334 [==============================] - 39s - loss: 3.3725 - acc: 0.1858 - val_loss: 3.4274 - val_acc: 0.1892
    Epoch 10/30
    333/334 [============================>.] - ETA: 0s - loss: 3.3099 - acc: 0.2000Epoch 00009: val_loss improved from 3.42739 to 3.29242, saving model to saved_models/weights.best.from_scratch.hdf5
    334/334 [==============================] - 39s - loss: 3.3101 - acc: 0.1997 - val_loss: 3.2924 - val_acc: 0.2132
    Epoch 11/30
    333/334 [============================>.] - ETA: 0s - loss: 3.2094 - acc: 0.2177Epoch 00010: val_loss did not improve
    334/334 [==============================] - 39s - loss: 3.2092 - acc: 0.2177 - val_loss: 3.3008 - val_acc: 0.2323
    Epoch 12/30
    333/334 [============================>.] - ETA: 0s - loss: 3.0734 - acc: 0.2380Epoch 00011: val_loss did not improve
    334/334 [==============================] - 40s - loss: 3.0725 - acc: 0.2382 - val_loss: 4.0531 - val_acc: 0.1401
    Epoch 13/30
    333/334 [============================>.] - ETA: 0s - loss: 3.0281 - acc: 0.2527Epoch 00012: val_loss improved from 3.29242 to 3.02524, saving model to saved_models/weights.best.from_scratch.hdf5
    334/334 [==============================] - 39s - loss: 3.0273 - acc: 0.2533 - val_loss: 3.0252 - val_acc: 0.2192
    Epoch 14/30
    333/334 [============================>.] - ETA: 0s - loss: 2.9584 - acc: 0.2629Epoch 00013: val_loss improved from 3.02524 to 3.00549, saving model to saved_models/weights.best.from_scratch.hdf5
    334/334 [==============================] - 39s - loss: 2.9584 - acc: 0.2627 - val_loss: 3.0055 - val_acc: 0.2395
    Epoch 15/30
    333/334 [============================>.] - ETA: 0s - loss: 2.8638 - acc: 0.2746Epoch 00014: val_loss did not improve
    334/334 [==============================] - 39s - loss: 2.8658 - acc: 0.2744 - val_loss: 3.0870 - val_acc: 0.2491
    Epoch 16/30
    333/334 [============================>.] - ETA: 0s - loss: 2.7892 - acc: 0.2883Epoch 00015: val_loss improved from 3.00549 to 2.78594, saving model to saved_models/weights.best.from_scratch.hdf5
    334/334 [==============================] - 39s - loss: 2.7895 - acc: 0.2879 - val_loss: 2.7859 - val_acc: 0.3054
    Epoch 17/30
    333/334 [============================>.] - ETA: 0s - loss: 2.7400 - acc: 0.2968Epoch 00016: val_loss improved from 2.78594 to 2.78523, saving model to saved_models/weights.best.from_scratch.hdf5
    334/334 [==============================] - 39s - loss: 2.7407 - acc: 0.2969 - val_loss: 2.7852 - val_acc: 0.3353
    Epoch 18/30
    333/334 [============================>.] - ETA: 0s - loss: 2.6714 - acc: 0.3177Epoch 00017: val_loss improved from 2.78523 to 2.65945, saving model to saved_models/weights.best.from_scratch.hdf5
    334/334 [==============================] - 39s - loss: 2.6702 - acc: 0.3178 - val_loss: 2.6594 - val_acc: 0.3138
    Epoch 19/30
    333/334 [============================>.] - ETA: 0s - loss: 2.6285 - acc: 0.3276Epoch 00018: val_loss improved from 2.65945 to 2.62756, saving model to saved_models/weights.best.from_scratch.hdf5
    334/334 [==============================] - 39s - loss: 2.6274 - acc: 0.3281 - val_loss: 2.6276 - val_acc: 0.3509
    Epoch 20/30
    333/334 [============================>.] - ETA: 0s - loss: 2.5731 - acc: 0.3339Epoch 00019: val_loss did not improve
    334/334 [==============================] - 39s - loss: 2.5742 - acc: 0.3337 - val_loss: 2.6537 - val_acc: 0.3186
    Epoch 21/30
    333/334 [============================>.] - ETA: 0s - loss: 2.5413 - acc: 0.3383Epoch 00020: val_loss did not improve
    334/334 [==============================] - 39s - loss: 2.5405 - acc: 0.3385 - val_loss: 2.7203 - val_acc: 0.3305
    Epoch 22/30
    333/334 [============================>.] - ETA: 0s - loss: 2.4762 - acc: 0.3562Epoch 00021: val_loss did not improve
    334/334 [==============================] - 39s - loss: 2.4768 - acc: 0.3561 - val_loss: 2.6754 - val_acc: 0.3198
    Epoch 23/30
    333/334 [============================>.] - ETA: 0s - loss: 2.4338 - acc: 0.3617Epoch 00022: val_loss improved from 2.62756 to 2.50800, saving model to saved_models/weights.best.from_scratch.hdf5
    334/334 [==============================] - 39s - loss: 2.4341 - acc: 0.3614 - val_loss: 2.5080 - val_acc: 0.3533
    Epoch 24/30
    333/334 [============================>.] - ETA: 0s - loss: 2.3895 - acc: 0.3736Epoch 00023: val_loss did not improve
    334/334 [==============================] - 39s - loss: 2.3907 - acc: 0.3731 - val_loss: 2.6018 - val_acc: 0.3569
    Epoch 25/30
    333/334 [============================>.] - ETA: 0s - loss: 2.3538 - acc: 0.3818Epoch 00024: val_loss did not improve
    334/334 [==============================] - 39s - loss: 2.3547 - acc: 0.3816 - val_loss: 2.7923 - val_acc: 0.3401
    Epoch 26/30
    333/334 [============================>.] - ETA: 0s - loss: 2.2934 - acc: 0.3995Epoch 00025: val_loss did not improve
    334/334 [==============================] - 39s - loss: 2.2933 - acc: 0.3997 - val_loss: 2.6003 - val_acc: 0.3389
    Epoch 27/30
    333/334 [============================>.] - ETA: 0s - loss: 2.2658 - acc: 0.3986Epoch 00026: val_loss did not improve
    334/334 [==============================] - 39s - loss: 2.2654 - acc: 0.3987 - val_loss: 2.5604 - val_acc: 0.3713
    Epoch 28/30
    333/334 [============================>.] - ETA: 0s - loss: 2.2220 - acc: 0.4123Epoch 00027: val_loss did not improve
    334/334 [==============================] - 39s - loss: 2.2213 - acc: 0.4123 - val_loss: 2.6464 - val_acc: 0.3545
    Epoch 29/30
    333/334 [============================>.] - ETA: 0s - loss: 2.2124 - acc: 0.4096Epoch 00028: val_loss improved from 2.50800 to 2.39041, saving model to saved_models/weights.best.from_scratch.hdf5
    334/334 [==============================] - 39s - loss: 2.2124 - acc: 0.4096 - val_loss: 2.3904 - val_acc: 0.3976
    Epoch 30/30
    333/334 [============================>.] - ETA: 0s - loss: 2.1680 - acc: 0.4248Epoch 00029: val_loss improved from 2.39041 to 2.32198, saving model to saved_models/weights.best.from_scratch.hdf5
    334/334 [==============================] - 39s - loss: 2.1669 - acc: 0.4249 - val_loss: 2.3220 - val_acc: 0.4132



```python
utility.plot_accuracy(history)
utility.plot_loss(history)
```


![png](output_35_0.png)



![png](output_35_1.png)


### Load the Model with the Best Validation Loss


```python
model.load_weights('saved_models/weights.best.from_scratch.hdf5')
```

### Test the Model

Try out your model on the test dataset of dog images.  Ensure that your test accuracy is greater than 1%.


```python
# get index of predicted dog breed for each image in test set
dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```

    Test accuracy: 38.3971%


---
<a id='step4'></a>
## Step 4: Use a CNN to Classify Dog Breeds

To reduce training time without sacrificing accuracy, we show you how to train a CNN using transfer learning.  In the following step, you will get a chance to use transfer learning to train your own CNN.

### Obtain Bottleneck Features


```python
# Load pre-computed bottleneck features
bottleneck_features = np.load('bottleneck_features/DogVGG16Data.npz')
train_VGG16 = bottleneck_features['train']
valid_VGG16 = bottleneck_features['valid']
test_VGG16 = bottleneck_features['test']
```

### Model Architecture

The model uses the the pre-trained VGG-16 model as a fixed feature extractor, where the last convolutional output of VGG-16 is fed as input to our model.  We only add a global average pooling layer and a fully connected layer, where the latter contains one node for each dog category and is equipped with a softmax.


```python
VGG16_model = Sequential()
VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
VGG16_model.add(Dense(133, activation='softmax'))

VGG16_model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    global_average_pooling2d_3 ( (None, 512)               0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 133)               68229     
    =================================================================
    Total params: 68,229.0
    Trainable params: 68,229.0
    Non-trainable params: 0.0
    _________________________________________________________________


### Compile the Model


```python
VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
```

### Train the Model


```python
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG16.hdf5', 
                               verbose=1, save_best_only=True)

VGG16_history = VGG16_model.fit(train_VGG16, train_targets, 
          validation_data=(valid_VGG16, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)
```

    Train on 6680 samples, validate on 835 samples
    Epoch 1/20
    6460/6680 [============================>.] - ETA: 0s - loss: 12.7266 - acc: 0.1133Epoch 00000: val_loss improved from inf to 11.49891, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s - loss: 12.6793 - acc: 0.1157 - val_loss: 11.4989 - val_acc: 0.1952
    Epoch 2/20
    6560/6680 [============================>.] - ETA: 0s - loss: 10.8418 - acc: 0.2537Epoch 00001: val_loss improved from 11.49891 to 10.65139, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 10.8436 - acc: 0.2534 - val_loss: 10.6514 - val_acc: 0.2707
    Epoch 3/20
    6440/6680 [===========================>..] - ETA: 0s - loss: 10.2766 - acc: 0.3132Epoch 00002: val_loss improved from 10.65139 to 10.44654, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 10.2829 - acc: 0.3127 - val_loss: 10.4465 - val_acc: 0.2946
    Epoch 4/20
    6620/6680 [============================>.] - ETA: 0s - loss: 10.1084 - acc: 0.3394Epoch 00003: val_loss improved from 10.44654 to 10.33140, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 10.0973 - acc: 0.3404 - val_loss: 10.3314 - val_acc: 0.3054
    Epoch 5/20
    6620/6680 [============================>.] - ETA: 0s - loss: 9.9235 - acc: 0.3600Epoch 00004: val_loss improved from 10.33140 to 10.19621, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 9.9183 - acc: 0.3600 - val_loss: 10.1962 - val_acc: 0.3186
    Epoch 6/20
    6560/6680 [============================>.] - ETA: 0s - loss: 9.8103 - acc: 0.3694Epoch 00005: val_loss improved from 10.19621 to 10.12693, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 9.8178 - acc: 0.3692 - val_loss: 10.1269 - val_acc: 0.3246
    Epoch 7/20
    6480/6680 [============================>.] - ETA: 0s - loss: 9.7595 - acc: 0.3779Epoch 00006: val_loss improved from 10.12693 to 10.06961, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 9.7307 - acc: 0.3801 - val_loss: 10.0696 - val_acc: 0.3293
    Epoch 8/20
    6540/6680 [============================>.] - ETA: 0s - loss: 9.5912 - acc: 0.3855Epoch 00007: val_loss improved from 10.06961 to 9.93621, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 9.6031 - acc: 0.3849 - val_loss: 9.9362 - val_acc: 0.3353
    Epoch 9/20
    6460/6680 [============================>.] - ETA: 0s - loss: 9.3997 - acc: 0.3950Epoch 00008: val_loss improved from 9.93621 to 9.86013, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 9.3907 - acc: 0.3955 - val_loss: 9.8601 - val_acc: 0.3281
    Epoch 10/20
    6640/6680 [============================>.] - ETA: 0s - loss: 9.1699 - acc: 0.4077Epoch 00009: val_loss improved from 9.86013 to 9.62476, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 9.1637 - acc: 0.4081 - val_loss: 9.6248 - val_acc: 0.3413
    Epoch 11/20
    6480/6680 [============================>.] - ETA: 0s - loss: 9.0187 - acc: 0.4239Epoch 00010: val_loss improved from 9.62476 to 9.60083, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 9.0189 - acc: 0.4240 - val_loss: 9.6008 - val_acc: 0.3545
    Epoch 12/20
    6620/6680 [============================>.] - ETA: 0s - loss: 8.9492 - acc: 0.4311Epoch 00011: val_loss improved from 9.60083 to 9.49035, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 8.9473 - acc: 0.4313 - val_loss: 9.4904 - val_acc: 0.3497
    Epoch 13/20
    6640/6680 [============================>.] - ETA: 0s - loss: 8.8213 - acc: 0.4413Epoch 00012: val_loss improved from 9.49035 to 9.40974, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 8.8194 - acc: 0.4413 - val_loss: 9.4097 - val_acc: 0.3641
    Epoch 14/20
    6580/6680 [============================>.] - ETA: 0s - loss: 8.8078 - acc: 0.4464Epoch 00013: val_loss improved from 9.40974 to 9.34264, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 8.7902 - acc: 0.4472 - val_loss: 9.3426 - val_acc: 0.3665
    Epoch 15/20
    6460/6680 [============================>.] - ETA: 0s - loss: 8.6445 - acc: 0.4511Epoch 00014: val_loss improved from 9.34264 to 9.16963, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 8.6582 - acc: 0.4503 - val_loss: 9.1696 - val_acc: 0.3737
    Epoch 16/20
    6420/6680 [===========================>..] - ETA: 0s - loss: 8.4433 - acc: 0.4621Epoch 00015: val_loss improved from 9.16963 to 8.96086, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 8.4201 - acc: 0.4635 - val_loss: 8.9609 - val_acc: 0.3772
    Epoch 17/20
    6460/6680 [============================>.] - ETA: 0s - loss: 8.1667 - acc: 0.4822Epoch 00016: val_loss improved from 8.96086 to 8.76480, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 8.1923 - acc: 0.4805 - val_loss: 8.7648 - val_acc: 0.3952
    Epoch 18/20
    6520/6680 [============================>.] - ETA: 0s - loss: 8.1133 - acc: 0.4885Epoch 00017: val_loss did not improve
    6680/6680 [==============================] - 1s - loss: 8.1072 - acc: 0.4891 - val_loss: 8.8091 - val_acc: 0.4036
    Epoch 19/20
    6500/6680 [============================>.] - ETA: 0s - loss: 8.0330 - acc: 0.4960Epoch 00018: val_loss improved from 8.76480 to 8.74885, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 8.0703 - acc: 0.4937 - val_loss: 8.7489 - val_acc: 0.4012
    Epoch 20/20
    6420/6680 [===========================>..] - ETA: 0s - loss: 8.0312 - acc: 0.4950Epoch 00019: val_loss improved from 8.74885 to 8.68178, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 8.0287 - acc: 0.4954 - val_loss: 8.6818 - val_acc: 0.4072



```python
utility.plot_accuracy(VGG16_history)
utility.plot_loss(VGG16_history)
```


![png](output_48_0.png)



![png](output_48_1.png)


### Load the Model with the Best Validation Loss


```python
VGG16_model.load_weights('saved_models/weights.best.VGG16.hdf5')
```

### Test the Model

Now, we can use the CNN to test how well it identifies breed within our test dataset of dog images.  We print the test accuracy below.


```python
# get index of predicted dog breed for each image in test set
VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG16]

# report test accuracy
test_accuracy = 100*np.sum(np.array(VGG16_predictions)==np.argmax(test_targets, axis=1))/len(VGG16_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```

    Test accuracy: 41.8660%


### Predict Dog Breed with the Model


```python
from extract_bottleneck_features import *

def VGG16_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_VGG16(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = VGG16_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]
```

---
<a id='step5'></a>
## Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)

You will now use transfer learning to create a CNN that can identify dog breed from images.  Your CNN must attain at least 60% accuracy on the test set.

In Step 4, we used transfer learning to create a CNN using VGG-16 bottleneck features.  In this section, you must use the bottleneck features from a different pre-trained model.  To make things easier for you, we have pre-computed the features for all of the networks that are currently available in Keras:
- [VGG-19](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG19Data.npz) bottleneck features
- [ResNet-50](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz) bottleneck features
- [Inception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz) bottleneck features
- [Xception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz) bottleneck features

The files are encoded as such:

    Dog{network}Data.npz
    
where `{network}`, in the above filename, can be one of `VGG19`, `Resnet50`, `InceptionV3`, or `Xception`.  Pick one of the above architectures, download the corresponding bottleneck features, and store the downloaded file in the `bottleneck_features/` folder in the repository.

### (IMPLEMENTATION) Obtain Bottleneck Features

In the code block below, extract the bottleneck features corresponding to the train, test, and validation sets by running the following:

    bottleneck_features = np.load('bottleneck_features/Dog{network}Data.npz')
    train_{network} = bottleneck_features['train']
    valid_{network} = bottleneck_features['valid']
    test_{network} = bottleneck_features['test']


```python
### Obtain bottleneck features from another pre-trained CNN.
bottleneck_features = np.load('bottleneck_features/DogInceptionV3Data.npz')

train_InceptionV3 = bottleneck_features['train']
valid_InceptionV3 = bottleneck_features['valid']
test_InceptionV3 = bottleneck_features['test']
```


```python
print(train_InceptionV3.shape[1:])
```

    (5, 5, 2048)


### (IMPLEMENTATION) Model Architecture

Create a CNN to classify dog breed.  At the end of your code cell block, summarize the layers of your model by executing the line:
    
        <your model's name>.summary()
   
__Question 5:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  Describe why you think the architecture is suitable for the current problem.

__Answer:__ 
The selected InceptionV3 network is already highly optimized for image detection. Adding additional convolution layers did not increase accuracy. 

Overfitting appears to be an issue as validation accuracy flatlines while training accuracy continues to increase after only a few epochs.

In addition to regulaization and dropout, data augmentation was used (refer [Step 5 Alternative Implmentation Notebook](./dog_app_augmented.ipynb) to mitigate overfittting, but in this instance did not significantly increase accuracy.

This model acheives around 80% accuracy on the test set. As the current problem (diferentiating dog breeds) is quite difficult, this result is currently acceptable.

### Note: For an alternative implementaton that uses Data Augmentation with Transfer Learning refer this notebook - [Step 5 Alternative Implmentation Notebook](./dog_app_augmented.ipynb)


```python
### Define your architecture.
InceptionV3_model = Sequential()
InceptionV3_model.add(Conv2D(filters=96, kernel_size=2, strides=1, padding='valid', input_shape=train_InceptionV3.shape[1:], activation='relu'))
InceptionV3_model.add(BatchNormalization(axis=1))
InceptionV3_model.add(Dropout(0.3))
InceptionV3_model.add(GlobalAveragePooling2D())
InceptionV3_model.add(Dropout(0.4))
InceptionV3_model.add(Dense(133, activation='softmax'))

InceptionV3_model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_13 (Conv2D)           (None, 4, 4, 96)          786528    
    _________________________________________________________________
    batch_normalization_13 (Batc (None, 4, 4, 96)          16        
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 4, 4, 96)          0         
    _________________________________________________________________
    global_average_pooling2d_4 ( (None, 96)                0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 96)                0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 133)               12901     
    =================================================================
    Total params: 799,445.0
    Trainable params: 799,437.0
    Non-trainable params: 8.0
    _________________________________________________________________



```python
# serialize model to JSON
model_json = InceptionV3_model.to_json()
with open("saved_models/inceptionv3_model.json", "w") as json_file:
    json_file.write(model_json)
```


```python
# test loading model from JSON
json_file = open('saved_models/inceptionv3_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_inceptionv3_model = model_from_json(loaded_model_json)

loaded_inceptionv3_model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_13 (Conv2D)           (None, 4, 4, 96)          786528    
    _________________________________________________________________
    batch_normalization_13 (Batc (None, 4, 4, 96)          16        
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 4, 4, 96)          0         
    _________________________________________________________________
    global_average_pooling2d_4 ( (None, 96)                0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 96)                0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 133)               12901     
    =================================================================
    Total params: 799,445.0
    Trainable params: 799,437.0
    Non-trainable params: 8.0
    _________________________________________________________________


### (IMPLEMENTATION) Compile the Model


```python
### Compile the model.
InceptionV3_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
```

### (IMPLEMENTATION) Train the Model

Train your model in the code cell below.  Use model checkpointing to save the model that attains the best validation loss.  

You are welcome to [augment the training data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html), but this is not a requirement. 


```python
### TODO: Train the model.
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.InceptionV3.hdf5', verbose=1, save_best_only=True)

InceptionV3_history = InceptionV3_model.fit(train_InceptionV3, train_targets, 
          validation_data=(valid_InceptionV3, valid_targets),
          epochs=10, batch_size=20, callbacks=[checkpointer], verbose=1)
```

    Train on 6680 samples, validate on 835 samples
    Epoch 1/10
    6600/6680 [============================>.] - ETA: 0s - loss: 2.3380 - acc: 0.5435Epoch 00000: val_loss improved from inf to 0.89963, saving model to saved_models/weights.best.InceptionV3.hdf5
    6680/6680 [==============================] - 4s - loss: 2.3249 - acc: 0.5467 - val_loss: 0.8996 - val_acc: 0.7701
    Epoch 2/10
    6640/6680 [============================>.] - ETA: 0s - loss: 1.0249 - acc: 0.7584Epoch 00001: val_loss improved from 0.89963 to 0.69354, saving model to saved_models/weights.best.InceptionV3.hdf5
    6680/6680 [==============================] - 3s - loss: 1.0235 - acc: 0.7585 - val_loss: 0.6935 - val_acc: 0.8048
    Epoch 3/10
    6660/6680 [============================>.] - ETA: 0s - loss: 0.7354 - acc: 0.8044Epoch 00002: val_loss improved from 0.69354 to 0.61427, saving model to saved_models/weights.best.InceptionV3.hdf5
    6680/6680 [==============================] - 3s - loss: 0.7344 - acc: 0.8045 - val_loss: 0.6143 - val_acc: 0.8108
    Epoch 4/10
    6660/6680 [============================>.] - ETA: 0s - loss: 0.6141 - acc: 0.8267Epoch 00003: val_loss improved from 0.61427 to 0.58891, saving model to saved_models/weights.best.InceptionV3.hdf5
    6680/6680 [==============================] - 3s - loss: 0.6142 - acc: 0.8269 - val_loss: 0.5889 - val_acc: 0.8359
    Epoch 5/10
    6640/6680 [============================>.] - ETA: 0s - loss: 0.5271 - acc: 0.8462Epoch 00004: val_loss improved from 0.58891 to 0.57575, saving model to saved_models/weights.best.InceptionV3.hdf5
    6680/6680 [==============================] - 3s - loss: 0.5275 - acc: 0.8463 - val_loss: 0.5757 - val_acc: 0.8323
    Epoch 6/10
    6640/6680 [============================>.] - ETA: 0s - loss: 0.4829 - acc: 0.8578Epoch 00005: val_loss improved from 0.57575 to 0.56571, saving model to saved_models/weights.best.InceptionV3.hdf5
    6680/6680 [==============================] - 3s - loss: 0.4830 - acc: 0.8578 - val_loss: 0.5657 - val_acc: 0.8228
    Epoch 7/10
    6600/6680 [============================>.] - ETA: 0s - loss: 0.4352 - acc: 0.8703Epoch 00006: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.4372 - acc: 0.8699 - val_loss: 0.5718 - val_acc: 0.8371
    Epoch 8/10
    6640/6680 [============================>.] - ETA: 0s - loss: 0.4000 - acc: 0.8792Epoch 00007: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.4004 - acc: 0.8789 - val_loss: 0.5686 - val_acc: 0.8407
    Epoch 9/10
    6620/6680 [============================>.] - ETA: 0s - loss: 0.3661 - acc: 0.8875Epoch 00008: val_loss improved from 0.56571 to 0.56414, saving model to saved_models/weights.best.InceptionV3.hdf5
    6680/6680 [==============================] - 3s - loss: 0.3673 - acc: 0.8871 - val_loss: 0.5641 - val_acc: 0.8503
    Epoch 10/10
    6600/6680 [============================>.] - ETA: 0s - loss: 0.3208 - acc: 0.9011Epoch 00009: val_loss improved from 0.56414 to 0.56154, saving model to saved_models/weights.best.InceptionV3.hdf5
    6680/6680 [==============================] - 3s - loss: 0.3204 - acc: 0.9010 - val_loss: 0.5615 - val_acc: 0.8527



```python
utility.plot_accuracy(InceptionV3_history)
utility.plot_loss(InceptionV3_history)
```


![png](output_67_0.png)



![png](output_67_1.png)


### (IMPLEMENTATION) Load the Model with the Best Validation Loss


```python
from keras.models import model_from_json

# load json and create model
json_file = open('saved_models/inceptionv3_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)

### Load the model weights with the best validation loss.
loaded_model.load_weights('saved_models/weights.best.InceptionV3.hdf5')
```

### (IMPLEMENTATION) Test the Model

Try out your model on the test dataset of dog images. Ensure that your test accuracy is greater than 60%.


```python
# get index of predicted dog breed for each image in test set
InceptionV3_predictions = [np.argmax(InceptionV3_model.predict(np.expand_dims(feature, axis=0))) for feature in test_InceptionV3]

# report test accuracy
test_accuracy = 100*np.sum(np.array(InceptionV3_predictions)==np.argmax(test_targets, axis=1))/len(InceptionV3_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```

    Test accuracy: 82.4163%


### (IMPLEMENTATION) Predict Dog Breed with the Model

Write a function that takes an image path as input and returns the dog breed (`Affenpinscher`, `Afghan_hound`, etc) that is predicted by your model.  

Similar to the analogous function in Step 5, your function should have three steps:
1. Extract the bottleneck features corresponding to the chosen CNN model.
2. Supply the bottleneck features as input to the model to return the predicted vector.  Note that the argmax of this prediction vector gives the index of the predicted dog breed.
3. Use the `dog_names` array defined in Step 0 of this notebook to return the corresponding breed.

The functions to extract the bottleneck features can be found in `extract_bottleneck_features.py`, and they have been imported in an earlier code cell.  To obtain the bottleneck features corresponding to your chosen CNN architecture, you need to use the function

    extract_{network}
    
where `{network}`, in the above filename, should be one of `VGG19`, `Resnet50`, `InceptionV3`, or `Xception`.


```python
### Returns the dog breed that is predicted by the model.
def InceptionV3_predict_breed(img_path): 
    # extract bottleneck features
    bottleneck_feature = extract_InceptionV3(path_to_tensor(img_path))
    
    # obtain predicted vector
    predicted_vector = InceptionV3_model.predict(bottleneck_feature)
    
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]
```

---
<a id='step6'></a>
## Step 6: Write your Algorithm

Write an algorithm that accepts a file path to an image and first determines whether the image contains a human, dog, or neither.  Then,
- if a __dog__ is detected in the image, return the predicted breed.
- if a __human__ is detected in the image, return the resembling dog breed.
- if __neither__ is detected in the image, provide output that indicates an error.

You are welcome to write your own functions for detecting humans and dogs in images, but feel free to use the `face_detector` and `dog_detector` functions developed above.  You are __required__ to use your CNN from Step 5 to predict dog breed.  

Some sample output for our algorithm is provided below, but feel free to design your own user experience!

![Sample Human Output](images/sample_human_output.png)


### (IMPLEMENTATION) Write your Algorithm


```python
def predict_dog_breed_from_human_or_dog(img_path):
    image_type = "error"
    dog_breed = "N/A"
    
    if face_detector(img_path):
        image_type = "face"
        dog_breed = InceptionV3_predict_breed(img_path)
    elif dog_detector(img_path):
        image_type = "dog"
        dog_breed = InceptionV3_predict_breed(img_path)
    
    return image_type, dog_breed
```

---
<a id='step7'></a>
## Step 7: Test Your Algorithm

In this section, you will take your new algorithm for a spin!  What kind of dog does the algorithm think that __you__ look like?  If you have a dog, does it predict your dog's breed accurately?  If you have a cat, does it mistakenly think that your cat is a dog?

### (IMPLEMENTATION) Test Your Algorithm on Sample Images!

Test your algorithm at least six images on your computer.  Feel free to use any images you like.  Use at least two human and two dog images.  

__Question 6:__ Is the output better than you expected :) ?  Or worse :( ?  Provide at least three possible points of improvement for your algorithm.

__Answer:__ 


```python
## TODO: Execute your algorithm from Step 6 on
## at least 6 images on your computer.
## Feel free to use as many code cells as needed.
import matplotlib.image as mpimg                     
%matplotlib inline 

def show_image(img_path):
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.show()
    
def test_algorithm(img_path):
    print("\nProcessing: {}".format(img_path))
    show_image(img_path)
    
    image_type, dog_breed = predict_dog_breed_from_human_or_dog(img_path)

    if image_type is "face":       
        print("This human looks like a {}".format(dog_breed))
    elif image_type is "dog":
        print("This dog looks like a {}".format(dog_breed))
    else:
        print("Error neither human or dog detected!")
```


```python
# load list of dog names
sample_images = glob("images/*")

for sample_image in sample_images:
    test_algorithm(sample_image)
```

    
    Processing: images/Curly-coated_retriever_03896.jpg



![png](output_78_1.png)


    This dog looks like a Curly-coated_retriever
    
    Processing: images/sample_human_output.png



![png](output_78_3.png)


    This human looks like a Lowchen
    
    Processing: images/American_water_spaniel_00648.jpg



![png](output_78_5.png)


    This dog looks like a American_water_spaniel
    
    Processing: images/Welsh_springer_spaniel_08203.jpg



![png](output_78_7.png)


    This dog looks like a Welsh_springer_spaniel
    
    Processing: images/Brittany_02625.jpg



![png](output_78_9.png)


    This dog looks like a Brittany
    
    Processing: images/Labrador_retriever_06455.jpg



![png](output_78_11.png)


    This dog looks like a Labrador_retriever
    
    Processing: images/Labrador_retriever_06457.jpg



![png](output_78_13.png)


    This dog looks like a Labrador_retriever
    
    Processing: images/sample_dog_output.png



![png](output_78_15.png)


    Error neither human or dog detected!
    
    Processing: images/sample_cnn.png



![png](output_78_17.png)


    Error neither human or dog detected!
    
    Processing: images/Labrador_retriever_06449.jpg



![png](output_78_19.png)


    This dog looks like a Labrador_retriever



```python

```
