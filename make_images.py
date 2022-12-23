from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


if __name__ == "__main__":

    
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    rand_image = x_train[np.random.randint(0, x_train.shape[0])]
    
    plt.subplot(1, 2, 1)
    plt.imshow(rand_image, cmap='gray')
    plt.axis('off')

    # Load the saved generator model
    generator = keras.models.load_model('models\\gan_mnist_generator_20.h5', compile=False)

    # Generate some noise as input for the generator
    # noise = np.random.rand(10, 100)

    # Use the generator to create fake images
    # fake_images = generator.predict(noise)
    fake_image = generator.predict(rand_image)

    # Rescale the images from [-1, 1] to [0, 255]
    # fake_images = (fake_images + 1) * 127.5
    fake_image = (fake_image + 1) * 127.5

    plt.sublot(1, 2, 2)
    plt.imshow(fake_image.reshape(28, 28), cmap='gray')
    plt.axis('off')
    
    plt.show()

    # # Plot the fake images
    # for i in range(10):
    #     plt.subplot(2, 5, i+1)
    #     plt.imshow(fake_image.reshape(28, 28), cmap='gray')
    #     plt.axis('off')
    # plt.show()
    # print()
