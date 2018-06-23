# import Python libraries
import requests
from glob import glob

DOG_API_URL = "http://localhost:5000/get_dog_breed"


def process_folder(folder_path):
    """send each image in folder to server for processing"""
    sample_images = glob(folder_path + "/*")

    for img_path in sample_images:
        image = open(img_path, "rb").read()
        payload = {"image": image}
        result = requests.post(DOG_API_URL, files=payload).json()

        if result["success"]:
            if result["is_human"]:
                print("{} is a human that looks like a {}".format(
                    img_path, result["dog_breed"]))
            elif result["is_dog"]:
                print("{} is a dog that looks like a {}".format(
                    img_path, result["dog_breed"]))
            else:
                print("{} is not a human or dog!".format(img_path))

        else:
            print("Request failed")


if __name__ == "__main__":
    process_folder("images")
