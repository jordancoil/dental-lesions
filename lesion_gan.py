import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms, datasets

from skimage.color import rgb2gray
import skimage.transform as sktrans

from utils import Logger
import matplotlib.pyplot as plt

from lesion_dataset_for_cGAN import LesionDatasetCGAN

import sys, argparse
import decimal
import random
import pandas as pd


class CustomResize(object):
    
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        return sktrans.resize(image, self.output_size)


class CustomGrayscale(object):

    def __call__(self, image):
        return rgb2gray(image)

def lesion_data():
    custom_transforms = transforms.Compose([
        # CustomGrayscale(),
        CustomResize((64,64))
    ])
    df = pd.read_csv("./data_csvs/cGAN_data.csv")
    return LesionDatasetCGAN(df, "./lesion_images/all_images_processed_3/", transform=custom_transforms)
    

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
        - A Sigmoid function is applied to the output 
        to obtain a value in the range(0, 1)
    """
    def __init__(self, image__size, num_channels):
        super(DiscriminatorNet, self).__init__()

        # eg. 28x28 = 784 (image vector size)
        n_features = image_size         
        n_out = 1
        n_channels = num_channels

        # input: (n_channels) x 64 x 64
        self.layer_0 = nn.Sequential(
            nn.Conv2d(n_channels, n_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # state size: (n_features) x 32 x 32
        self.layer_1 = nn.Sequential(
            nn.Conv2d(n_features, n_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # state size: (n_featuers * 2) x 16 x 16
        self.layer_2 = nn.Sequential(
            nn.Conv2d(n_features * 2, n_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # state size: (n_featuers * 4) x 8 x 8
        self.layer_3 = nn.Sequential(
            nn.Conv2d(n_features * 4, n_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.out = nn.Sequential(
            nn.Conv2d(n_features * 8, n_out, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer_0(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
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
            LeakyReLU avoids vanishing gradient problem
        - A TanH function is applied to the output to
        map resulting values into (-1, 1) range (same
        range as MNIST)
    """
    def __init__(self, image_size, num_channels, latent_vector_size):
        super(GeneratorNet, self).__init__()
        n_features = latent_vector_size
        n_out = image_size # eg. 28x28 = 784 (image size)
        n_channels = num_channels

        # Naming the arguments are a guide for the other layers.
        self.layer_0 = nn.Sequential(
            nn.ConvTranspose2d(n_features, n_out * 8, 
                kernel_size=4, 
                stride=1, 
                padding=0, 
                bias=False),
            nn.BatchNorm2d(n_out * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # state size. (n_out*8) x 4 x 4

        self.layer_1 = nn.Sequential(
            nn.ConvTranspose2d(n_out * 8, n_out * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_out * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # state size. (n_out*4) x 8 x 8

        self.layer_2 = nn.Sequential(
            nn.ConvTranspose2d(n_out * 4, n_out * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_out * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # state size. (n_out*2) x 16 x 16

        self.layer_3 = nn.Sequential(
            nn.ConvTranspose2d(n_out * 2, n_out, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_out),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # state size. (n_out) x 32 x 32

        self.out = nn.Sequential(
            nn.ConvTranspose2d(n_out, n_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        # state size. (n_channels) x 64 x 64

    def forward(self, x):
        x = self.layer_0(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.out(x)
        return x



def images_to_vectors(images, target_size):
    # helper function to flatten images
    return images.view(images.size(0), target_size)

def vectors_to_images(vectors, target_size):
    # helper function to unflatten images
    return vectors.view(vectors.size(0), 1, target_size[0], target_size[1])

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
def ones_target(size, noisy):
    """
    Tensor containing ones, with shape = size
    """
    if noisy:
        data = Variable(torch.Tensor(size, 1, 1, 1))
        random_number = float(decimal.Decimal(random.randrange(900, 1000))/1000)
        data.fill_(random_number)
    else:
        data.fill_(random_number)
    return data

def zeros_target(size, noisy):
    """
    Tensor containing zeros, with shape = size
    """
    if noisy:
        data = Variable(torch.Tensor(size, 1, 1, 1))
        random_number = float(decimal.Decimal(random.randrange(0, 100))/1000)
        data.fill_(random_number)
    else:
        data = Variable(torch.zeros(size, 1, 1, 1))
    return data


def train_discriminator(Discriminator, optimizer, real_data, fake_data):
    N = real_data.size(0)

    # Reset gradients
    optimizer.zero_grad()

    # 1.1 Train on Real Data
    prediction_real = Discriminator(real_data)
    # calculate error and backpropogate
    error_real = loss(prediction_real, ones_target(N, noisy=True))
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = Discriminator(fake_data)
    # Calculate error and backpropogate
    error_fake = loss(prediction_fake, zeros_target(N, noisy=True))
    error_fake.backward()

    # 1.3 Update weights with gradients
    optimizer.step()

    # Return error and prediction for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake


def train_generator(Discriminator, optimizer, fake_data):
    N = fake_data.size(0)

    # Reset gradients
    optimizer.zero_grad()

    prediction = Discriminator(fake_data)

    # Calculate error and backpropogate
    error = loss(prediction, ones_target(N, noisy=True))
    error.backward()

    # Update weights with gradients
    optimizer.step()

    return error


def weights_init(model):
    """
    The DCGAN paper specifies that the weights should be randomly initialized
    from a Normal dist of mean=0, stdv=0.02
    """
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

def plot_losses(g_losses, d_losses):
    # Plot losses after training
    # TODO: Move to logger class
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="Gen")
    plt.plot(d_losses, label="Dis")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Lesion GAN')
    parser.add_argument('--xfile', type=str, nargs='?')
    parser.add_argument('--test', dest='feature', action='store_true')
    options = parser.parse_args()

    # Load Data
    data = lesion_data()
    image_size = (224, 224)
    vector_size = image_size[0] * image_size[1]

    # DCGAN Variables
    workers = 2
    batch_size = 16
    image_size = 64 # 64 x 64 x num_channels
    num_channels = 1
    latent_vector_size = 100
    num_feature_maps = 64 # size of feature maps in D and G
    lr_d = 0.0005 # Learning Rate for optimizers
    lr_g = 2e-4
    beta1 = 0.5 # beta1 hyperparam for Adam optim.

    # Create loader to iterate over data
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, num_workers=workers, shuffle=True)
    num_batches = len(data_loader)

    # Initialize discriminator and generator and initialize weights
    Discriminator = DiscriminatorNet(image_size, num_channels)
    Generator = GeneratorNet(image_size, num_channels, latent_vector_size)
    Generator.apply(weights_init)
    Discriminator.apply(weights_init)

    d_optimizer = optim.Adam(Discriminator.parameters(), lr=lr_d, betas=(beta1, 0.999))
    g_optimizer = optim.Adam(Generator.parameters(), lr=lr_g, betas=(beta1, 0.999))

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
    fixed_noise = torch.randn(num_test_samples, latent_vector_size, 1, 1)

    # START training code

    # Create logger instance
    logger = Logger(model_name='VGAN', data_name='LESION')
    g_losses = []
    d_losses = []

    num_epochs = 200

    try:
        for epoch in range(num_epochs):
            for n_batch, (real_batch, lesion, tooth_num) in enumerate(data_loader):
                N = real_batch.size(0)

                # 1. Train Discriminator
                #real_data = Variable(images_to_vectors(real_batch, vector_size))
                real_data = real_batch

                # Generate fake data and detach
                # (so gradients are not calculated for generator)
                gen_noise = torch.randn(N, 100, 1, 1)
                fake_data = Generator(gen_noise).detach()
                #fake_data = generator(noise(N)).detach()

                # Train Discrimiator
                d_error, d_pred_real, d_pred_fake = \
                    train_discriminator(Discriminator, d_optimizer, real_data, fake_data)


                # 2. Train Generator
                #fake_data = generator(noise(N))
                fake_data = Generator(gen_noise)

                # Train Generator
                g_error = train_generator(Discriminator, g_optimizer, fake_data)


                # 3. Log Batch Error
                logger.log(d_error, g_error, epoch, n_batch, num_batches)

                # Save Losses for plotting later
                g_losses.append(g_error.item())
                d_losses.append(d_error.item())

                # 4. Display Progress periodically
                if (n_batch) % 10 == 0:
                    test_images = Generator(fixed_noise)
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
    except KeyboardInterrupt:
        # plot losses if keyboard interrupt
        plot_losses(g_losses, d_losses)

    # END training code
    plot_losses(g_losses, d_losses)

    # --- End tutorial code ---

    main(options)

