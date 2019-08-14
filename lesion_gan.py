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

from Networks.gan_networks import CGAN_DiscriminatorNet, CGAN_GeneratorNet

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
    #df = pd.read_csv("./data_csvs/cGAN_data.csv")
    #folder = "./lesion_images/all_images_processed_3/"
    df = pd.read_csv("./data_csvs/cGAN_data_subset_1.csv")
    folder = "./lesion_images/processed_3_zeros_only/type1/upwards/"
    return LesionDatasetCGAN(df, folder, transform=custom_transforms)

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


def train_discriminator(Discriminator, optimizer, real_data, labels, fake_data, fake_labels):
    N = real_data.size(0)

    # Reset gradients
    optimizer.zero_grad()

    # 1.1 Train on Real Data
    prediction_real = Discriminator(real_data, labels)
    # calculate error and backpropogate
    error_real = loss(prediction_real, ones_target(N, noisy=True))
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = Discriminator(fake_data, fake_labels)
    # Calculate error and backpropogate
    error_fake = loss(prediction_fake, zeros_target(N, noisy=True))
    error_fake.backward()

    # 1.3 Update weights with gradients
    optimizer.step()

    # Return error and prediction for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake


def train_generator(Discriminator, optimizer, fake_data, fake_labels):
    N = fake_data.size(0)

    # Reset gradients
    optimizer.zero_grad()

    prediction = Discriminator(fake_data, fake_labels)

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
    parser.add_argument('--gpu', dest='cuda', action='store_true')
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
    num_labels = 2 # Number of additional labels used by the cGAN
    latent_vector_size = 100
    num_feature_maps = 64 # size of feature maps in D and G
    lr_d = 0.0005 # Learning Rate for optimizers
    lr_g = 2e-4
    beta1 = 0.5 # beta1 hyperparam for Adam optim.

    # Create loader to iterate over data
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, num_workers=workers, shuffle=True)
    num_batches = len(data_loader)

    # Initialize discriminator and generator and initialize weights
    Discriminator = CGAN_DiscriminatorNet(image_size, num_channels, num_labels)
    Generator = CGAN_GeneratorNet(image_size, num_channels, num_labels, latent_vector_size)
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
    fixed_label_1 = torch.FloatTensor(num_test_samples, 1).random_(0, 2)
    fixed_label_2 = torch.FloatTensor(num_test_samples, 1).random_(0, 33)
    fixed_labels = torch.cat([fixed_label_1, fixed_label_2], 1)

    # START training code

    # Create logger instance
    logger = Logger(model_name='VGAN', data_name='LESION')
    g_losses = []
    d_losses = []

    num_epochs = 200

    try:
        for epoch in range(num_epochs):
            for n_batch, (real_batch, labels) in enumerate(data_loader):
                N = real_batch.size(0)

                if options.cuda:
                    real_batch = real_batch.cuda()
                    labels = labels.cuda()

                # 1. Train Discriminator
                #real_data = Variable(images_to_vectors(real_batch, vector_size))
                real_data = real_batch.float()
                labels = labels.float()

                # Generate fake data and detach
                # (so gradients are not calculated for generator)
                gen_noise = torch.randn(N, 100, 1, 1)
                fake_label_1 = torch.FloatTensor(N, 1).random_(0, 2)
                fake_label_2 = torch.FloatTensor(N, 1).random_(0, 33)
                fake_labels = torch.cat([fake_label_1, fake_label_2], 1)

                fake_data = Generator(gen_noise, fake_labels).detach()

                # Train Discrimiator
                d_error, d_pred_real, d_pred_fake = \
                    train_discriminator(Discriminator, d_optimizer, real_data, labels, fake_data, fake_labels)


                # 2. Train Generator
                #fake_data = generator(noise(N))
                fake_data = Generator(gen_noise, fake_labels)

                # Train Generator
                g_error = train_generator(Discriminator, g_optimizer, fake_data, fake_labels)


                # 3. Log Batch Error
                logger.log(d_error, g_error, epoch, n_batch, num_batches)

                # Save Losses for plotting later
                g_losses.append(g_error.item())
                d_losses.append(d_error.item())

                # 4. Display Progress periodically
                if (n_batch) % 10 == 0:
                    test_images = Generator(fixed_noise, fixed_labels)
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

