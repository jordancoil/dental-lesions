import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms, datasets

from utils import Logger

import sys, argparse

def main(argv):
    x_file = options.xfile
    # TODO: use x_file to run a GAN on lesion images

def mnist_data():
    compose = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((.5, .5, .5), (.5, .5, .5))
            ])

    out_dir = './dataset'
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

class DiscriminatorNet(nn.Module):
    """
    A three hidden-layer discriminative neural network

    This network takes a flattened image as its input,
    and returns the probability of it belonging to the
    real dataset, or the synthetic dataset.

    Structure:
        - 3 hidden layers, each followed by leaky-ReLU 
        and dropout to prevent overfitting.
        - A Sigmoid function is applied to the output 
        to obtain a value in the range(0, 1)
    """
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features = 784 # eg. 28x28 = 784 (image size)
        n_out = 1

        self.hidden_0 = nn.Sequential(
            nn.Linear(n_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden_1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden_2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.out = nn.Sequential(
            nn.Linear(256, n_out),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden_0(x)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        x = self.out(x)
        return x


class GeneratorNet(nn.Module):
    """
    A three hidden-layer generative neural network

    This network takes a latent variable vector as input,
    and returns a flattened vector representation of an
    image.

    Structure:
        - 3 hidden layers, each followed by leaky-ReLU
        - A TanH function is applied to the output to
        map resulting values into (-1, 1) range (same
        range as MNIST)
    """
    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 100
        n_out = 784 # eg. 28x28 = 784 (image size)

        self.hidden_0 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2)
        )

        self.hidden_1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )

        self.hidden_2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )

        self.out = nn.Sequential(
            nn.Linear(1024, n_out),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden_0(x)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        x = self.out(x)
        return x

def images_to_vectors(images):
    # helper function to flatten images
    return images.view(images.size(0), 784)

def vectors_to_images(vectors):
    # helper function to unflatten images
    # output: greyscale 28x28 images
    return vectors.view(vectors.size(0), 1, 28, 28)

def noise(size):
    """
    Generates a 1-d vector of gaussian sampled random values
    """
    n = Variable(torch.randn(size, 100))
    return n

"""
Real-images targets are always ones, and fake-images targets
are always zeros. These helper functions help with this.
"""
def ones_target(size):
    """
    Tensor containing ones, with shape = size
    """
    data = Variable(torch.ones(size, 1))
    return data

def zeros_target(size):
    """
    Tensor containing zeros, with shape = size
    """
    data = Variable(torch.zeros(size, 1))
    return data


def train_discriminator(optimizer, real_data, fake_data):
    N = real_data.size(0)

    # Reset gradients
    optimizer.zero_grad()

    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    # calculate error and backpropogate
    error_real = loss(prediction_real, ones_target(N))
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate error and backpropogate
    error_fake = loss(prediction_fake, zeros_target(N))
    error_fake.backward()

    # 1.3 Update weights with gradients
    optimizer.step()

    # Return error and prediction for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake


def train_generator(optimizer, fake_data):
    N = fake_data.size(0)

    # Reset gradients
    optimizer.zero_grad()

    prediction = discriminator(fake_data)

    # Calculate error and backpropogate
    error = loss(prediction, ones_target(N))
    error.backward()

    # Update weights with gradients
    optimizer.step()

    return error


if __name__ == "__main__":	
    parser = argparse.ArgumentParser(description='Run Lesion GAN')
    parser.add_argument('--xfile', type=str, nargs='?')
    parser.add_argument('--test', dest='feature', action='store_true')
    options = parser.parse_args()

    # --- GAN Tutorial Code ---
    data = mnist_data() # Load Data

    # Create loader to iterate over data
    data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)    
    num_batches = len(data_loader)

    discriminator = DiscriminatorNet()
    generator = GeneratorNet()

    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

    """
    Binary Cross Entropy Loss is used because it resembles 
    the log-loss for both Generator and Discriminator
    """
    loss = nn.BCELoss()


    """
    Every few steps we will visualize a batch of images from
    noise to see how our training process is developing
    """
    num_test_samples = 16
    test_noise = noise(num_test_samples)

    # START training code
    
    # Create logger instance
    logger = Logger(model_name='VGAN', data_name='MNIST')

    num_epochs = 200

    for epoch in range(num_epochs):
        for n_batch, (real_batch,_) in enumerate(data_loader):
            N = real_batch.size(0)

            # 1. Train Discriminator
            real_data = Variable(images_to_vectors(real_batch))

            # Generate fake data and detach
            # (so gradients are not calculated for generator)
            fake_data = generator(noise(N)).detach()

            # Train Discrimiator
            d_error, d_pred_real, d_pred_fake = \
                train_discriminator(d_optimizer, real_data, fake_data)

            
            # 2. Train Generator
            fake_data = generator(noise(N))

            # Train Generator
            g_error = train_generator(g_optimizer, fake_data)

            
            # 3. Log Batch Error
            logger.log(d_error, g_error, epoch, n_batch, num_batches)

            # 4. Display Progress periodically
            if (n_batch) % 100 == 0:
                test_images = vectors_to_images(generator(test_noise))
                test_images = test_images.data

                logger.log_images(
                    test_images, num_test_samples, 
                    epoch, n_batch, num_batches
                )

                # Display status logs
                logger.display_status(
                    epoch, num_epochs, n_batch, num_batches,
                    d_error, g_error, d_pred_real, d_pred_fake
                )

    # END training code

    # --- End tutorial code ---
    
    main(options)

