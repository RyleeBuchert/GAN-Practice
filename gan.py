from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Build the generator
def build_generator(latent_dim, optimizer, img_rows, img_cols, channels):
    generator = keras.Sequential()
    
    generator.add(keras.layers.Dense(units=256, input_dim=latent_dim))
    generator.add(keras.layers.LeakyReLU(alpha=0.2))
    generator.add(keras.layers.BatchNormalization())
    generator.add(keras.layers.Dropout(rate=0.2))
    
    generator.add(keras.layers.Dense(units=512))
    generator.add(keras.layers.LeakyReLU(alpha=0.2))
    generator.add(keras.layers.BatchNormalization())
    generator.add(keras.layers.Dropout(rate=0.3))
    
    generator.add(keras.layers.Dense(units=1024))
    generator.add(keras.layers.LeakyReLU(alpha=0.2))
    generator.add(keras.layers.BatchNormalization())
    generator.add(keras.layers.Dropout(rate=0.3))
    
    generator.add(keras.layers.Dense(units=img_rows*img_cols*channels, activation='tanh'))
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    return generator


# Build the discriminator
def build_discriminator(optimizer, img_rows, img_cols, channels):
    discriminator = keras.Sequential()
    
    discriminator.add(keras.layers.Dense(units=1024, input_dim=img_rows*img_cols*channels))
    discriminator.add(keras.layers.LeakyReLU(alpha=0.2))
    discriminator.add(keras.layers.BatchNormalization())
    discriminator.add(keras.layers.Dropout(rate=0.2))
    
    discriminator.add(keras.layers.Dense(units=512))
    discriminator.add(keras.layers.LeakyReLU(alpha=0.2))
    discriminator.add(keras.layers.BatchNormalization())
    discriminator.add(keras.layers.Dropout(rate=0.2))
    
    discriminator.add(keras.layers.Dense(units=256))
    discriminator.add(keras.layers.LeakyReLU(alpha=0.2))
    discriminator.add(keras.layers.BatchNormalization())
    discriminator.add(keras.layers.Dropout(rate=0.2))
    
    discriminator.add(keras.layers.Dense(units=1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    return discriminator


# The generator takes noise as input and generates images
def build_gan(generator, discriminator, latent_dim, optimizer):
    gan_input = keras.layers.Input(shape=(latent_dim,))
    fake_image = generator(gan_input)
    gan_output = discriminator(fake_image)
    gan = keras.models.Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    return gan


# Define the training loop
def train(gan, generator, discriminator, x_train, latent_dim=100, batch_size=32, epochs=10, epoch_steps=100):
    for epoch in range(epochs):
        for batch in range(epoch_steps):
            # Generate noise as input for the generator
            noise = np.random.rand(batch_size, latent_dim)
            
            # Use the generator to create fake images
            fake_images = generator.predict(noise)

            # Get a random batch of real images
            real_images = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

            # Concatenate real and fake images
            images = np.concatenate((real_images, fake_images))

            # Create labels for real and fake images
            labels = np.concatenate((np.ones(batch_size), np.zeros(batch_size)))

            # Add noise to the labels
            labels += 0.05 * np.random.random(labels.shape)

            # Train the discriminator
            d_loss = discriminator.train_on_batch(images, labels)

            # Generate new noise
            noise = np.random.rand(batch_size, latent_dim)

            # Get fake labels (all ones)
            fake_labels = np.ones(batch_size)

            # Train the generator (via the GAN model, where the discriminator weights are frozen)
            g_loss = gan.train_on_batch(noise, fake_labels)

            # Print progress
            print(f'Epoch: {epoch + 1} \t Discriminator Loss: {d_loss} \t Generator Loss: {g_loss}')
            noise = np.random.normal(0, 1, size=(25, latent_dim))
        show_images(generator, noise, (5, 5), 28, 28, 1)


# Show generated images
def show_images(generator, noise, size_fig, img_rows, img_cols, channels):
    generated_images = generator.predict(noise)
    plt.figure(figsize=size_fig)
    
    for i, image in enumerate(generated_images):
        plt.subplot(size_fig[0], size_fig[1], i+1)
        if channels == 1:
            plt.imshow(image.reshape(img_rows, img_cols), cmap='gray')
        else:
            plt.imshow(image.reshape(img_rows, img_cols, channels))
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(2488)
    
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Rescale the images from [0, 255] to [-1, 1]
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_test = (x_test.astype(np.float32) - 127.5) / 127.5

    # Flatten the images
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    # Set the input dimensions
    latent_dim = 100
    img_rows, img_cols, channels = 28, 28, 1

    # Set the optimizer
    optimizer = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)

    # Build the generator
    generator = build_generator(latent_dim, optimizer, img_rows, img_cols, channels)

    # Compile the discriminator
    discriminator = build_discriminator(optimizer, img_rows, img_cols, channels)
    
    # Set the discriminator weights to be non-trainable in the GAN model
    discriminator.trainable = False

    # Build the GAN
    gan = build_gan(generator, discriminator, latent_dim, optimizer)

    # Set the training parameters
    batch_size = 32
    epochs = 10
    epoch_steps = 1000

    # Train the GAN
    train(gan, generator, discriminator, x_train, latent_dim, batch_size, epochs, epoch_steps)

    # Save the generator model
    generator.save(f'models\\gan_mnist_generator_{epochs}_v2.h5')
