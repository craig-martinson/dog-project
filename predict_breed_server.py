# uncomment to force tensorflow to use CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# import Python libraries
import numpy as np
import tensorflow as tf

import keras as keras
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils

import math
import utility
import cv2
from glob import glob
from PIL import Image
import numpy as np
import flask
import io

app = flask.Flask(__name__)


def face_detector(image):
    """returns "True" if face is detected in image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def predict_breed(tensor, model):
    """Returns the dog breed that is predicted by the model"""
    inceptionV3_bottleneck_features = InceptionV3(weights='imagenet', include_top=False).predict(
        keras.applications.inception_v3.preprocess_input(tensor))
    predicted_vector = model.predict(inceptionV3_bottleneck_features)
    return dog_names[np.argmax(predicted_vector)]


def load_model(model_path, weights_path):
    """ load json and create model and loads weights """
    json_file = open(model_path, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)

    # load the model weights with the best validation loss.
    model.load_weights(weights_path)

    return model


def image_to_tensor(img):
    """convert PIL image to 4D tensor with shape (1, 224, 224, 3)"""
    if img.mode != "RGB":
        img = img.convert("RGB")

    img = img.resize((224, 224))

    x = img_to_array(img)
    return np.expand_dims(x, axis=0)


@app.route("/get_dog_breed", methods=["POST"])
def get_dog_breed():
    """this method processes any requests to the /get_dog_breed endpoint:"""
    data = {"success": False}

    # need to destroys the current TF graph or second request will fail
    keras.backend.clear_session()

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image_request = flask.request.files["image"]

            # read the image in PIL format
            img = Image.open(io.BytesIO(image_request.read()))

            # preprocess the image and prepare it for classification
            img2 = image_to_tensor(img)

            # use cv2 to predict if image is human or not
            is_human = face_detector(np.asarray(img))

            # use resnet50 to predict if image is dog or not
            # never do this is produciton, loads new model every request!
            resnet50_model = ResNet50(weights='imagenet')
            prediction = np.argmax(resnet50_model.predict(img2))
            is_dog = (prediction <= 268) & (prediction >= 151)

            dog_breed = "unknown"

            if(is_human or is_dog):
                # use inceptionv3 to predict dog breed
                # never do this is produciton, loads new model every request!
                inceptionv3_model = load_model('saved_models/inceptionv3_model.json',
                                               'saved_models/weights.best.InceptionV3.hdf5')
                # get dog breed
                dog_breed = predict_breed(img2, inceptionv3_model)

            # return results
            data["is_human"] = True if is_human else False
            data["is_dog"] = True if is_dog else False
            data["dog_breed"] = dog_breed

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# initialise the global variables and start the server
if __name__ == "__main__":
    print(("* Initialising..."))

    # extract pre-trained face detector
    face_cascade = cv2.CascadeClassifier(
        'haarcascades/haarcascade_frontalface_alt.xml')

    # load list of dog names
    dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

    # ignore tensorflow warnings
    tf.logging.set_verbosity(tf.logging.ERROR)

    app.run()
