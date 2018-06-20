import matplotlib.pyplot as plt
import tensorflow as tf
import keras as keras


def print_elapsed_time(total_time):
    ''' Prints elapsed time in hh:mm:ss format
    '''
    hh = int(total_time / 3600)
    mm = int((total_time % 3600) / 60)
    ss = int((total_time % 3600) % 60)
    print(
        "\n** Total Elapsed Runtime: {:0>2}:{:0>2}:{:0>2}".format(hh, mm, ss))

# Visualize Loss History
# refer https://chrisalbon.com/deep_learning/keras/visualize_loss_history/ for details


def visualize_loss_history(history):
    # Get training and test loss histories
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)

    # Visualize loss history
    plt.plot(epoch_count, training_loss, 'r--')
    plt.plot(epoch_count, validation_loss, 'b-')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


# Visualize original and sugmented Images
# refer https://github.com/udacity/aind2-cnn for details

def print_versions():
    print("Tensorflow version: {}".format(tf.__version__))
    print("Keras version: {}".format(keras.__version__))


def visualize_augmented_images(training_data, datagen, image_count):
    # take subset of training data
    training_data_subset = training_data[:image_count]

    # visualize subset of training data
    fig = plt.figure(figsize=(20, 2))

    for i in range(0, len(training_data_subset)):
        ax = fig.add_subplot(1, image_count, i+1)
        ax.imshow(training_data_subset[i])

    fig.suptitle('Subset of Original Training Images', fontsize=20)
    plt.show()

    # visualize augmented images
    fig = plt.figure(figsize=(20, 2))
    for x_batch in datagen.flow(training_data_subset, batch_size=12):
        for i in range(0, image_count):
            ax = fig.add_subplot(1, image_count, i+1)
            ax.imshow(x_batch[i])
        fig.suptitle('Augmented Images', fontsize=20)
        plt.show()
        break
