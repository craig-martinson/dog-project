# uncomment to force tensorflow to use CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# import Python Libraries
import numpy as np
import tensorflow as tf
import keras as keras
from keras.models import model_from_json
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
import math
import utility
import cv2
from glob import glob


def face_detector(img_path):
    """returns "True" if face is detected in image"""
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def path_to_tensor(img_path):
    """convert RGB image to 4D tensor with shape (1, 224, 224, 3)"""
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)


def dog_detector(img_path):
    """Uses ResNet50 to detect if image contains a dog"""
    img = keras.applications.resnet50.preprocess_input(
        path_to_tensor(img_path))
    prediction = np.argmax(ResNet50_model.predict(img))

    return ((prediction <= 268) & (prediction >= 151))


def extract_bottleneck_features(tensor):
    """Returns the InceptionV3 bottleneck features"""
    return InceptionV3(weights='imagenet', include_top=False).predict(keras.applications.inception_v3.preprocess_input(tensor))


def predict_breed(img_path):
    """Returns the dog breed that is predicted by the model"""
    inceptionV3_bottleneck_features = extract_bottleneck_features(
        path_to_tensor(img_path))
    predicted_vector = model.predict(inceptionV3_bottleneck_features)
    return dog_names[np.argmax(predicted_vector)]


def predict_dog_breed_from_human_or_dog(img_path):
    """Returns a tuple of image with image type (person or dog)
    and the dog breed that is predicted by the model
    """
    image_type = "error"
    dog_breed = "N/A"

    if face_detector(img_path):
        image_type = "face"
        dog_breed = predict_breed(img_path)
    elif dog_detector(img_path):
        image_type = "dog"
        dog_breed = predict_breed(img_path)

    return image_type, dog_breed


def load_model(model_path, weights_path):
    """ load json and create model and loads weights """
    # load json and create model
    json_file = open(model_path, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)

    # load the model weights with the best validation loss.
    model.load_weights(weights_path)

    return model


def process_folder(folder_path):
    """ process all images in folder """
    sample_images = glob(folder_path + "/*")

    for img_path in sample_images:
        image_type, dog_breed = predict_dog_breed_from_human_or_dog(img_path)

        if image_type is "face":
            print("{} is a human that looks like a {}".format(img_path, dog_breed))
        elif image_type is "dog":
            print("{} is a dog that looks like a {}".format(img_path, dog_breed))
        else:
            print("{} is not a human or dog!".format(img_path))


if __name__ == "__main__":
    print(("* Initialising..."))

    # ignore tensorflow warnings
    tf.logging.set_verbosity(tf.logging.ERROR)

    # define ResNet50 model
    ResNet50_model = ResNet50(weights='imagenet')

    # extract pre-trained face detector
    face_cascade = cv2.CascadeClassifier(
        'haarcascades/haarcascade_frontalface_alt.xml')

    # load list of dog names
    dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

    # load json and create model
    model = load_model('saved_models/inceptionv3_model.json',
                       'saved_models/weights.best.InceptionV3.hdf5')

    # load list of dog names
    process_folder("images")
