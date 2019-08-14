import torch
from torch import nn

class CGAN_DiscriminatorNet(nn.Module):
    """
    A Discriminator Network as part of a Conditional GAN

    This network takes a batch of images and labels as its input
    and returns a probability whether or not the images belong
    to a target dataset or not.

    Structure:
        - labels are embedding via Linear neural network layer
        - images are processed through 4 convolutional layers
          each with Batch Normalization and LeakyReLU
        - output layer combines convoluted images and embeddings
          and makes a prediction via Sigmoid function in the
          range (0, 1)
    """

    def __init__(self, image_size, num_channels, num_labels):
        super(CGAN_DiscriminatorNet, self).__init__()

        self.image_size = image_size

        # Image Variables
        n_features = image_size
        n_channels = num_channels

        # Label Variables
        n_labels = num_labels
        embedding_size = 50

        # Misc Variables
        n_out = 1

        self.label_embedding = nn.Sequential(
            nn.Linear(n_labels, embedding_size),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer_0 = nn.Sequential(
            nn.Conv2d(n_channels, n_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer_1 = nn.Sequential(
            nn.Conv2d(n_features, n_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer_2 = nn.Sequential(
            nn.Conv2d(n_features * 2, n_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer_3 = nn.Sequential(
            nn.Conv2d(n_features * 4, n_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.out = nn.Sequential(
            nn.Linear(n_features*8*4*4 + embedding_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, n_out),
            nn.Sigmoid()
        )


    def forward(self, images, labels):
        batch_size = images.size(0)
        
        # First 4 Convolutional Layers for Image
        # Output will be of size: (self.image_size * 8) * 4 * 4
        x = self.layer_0(images)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = x.view(batch_size, self.image_size*8*4*4) # This allows us to concat our images with the label embeddings

        y = self.label_embedding(labels)
        x = torch.cat([x, y], 1)

        return self.out(x)
            

class CGAN_GeneratorNet(nn.Module):
    """
    A Generative Network as part of a Conditional GAN

    This network takes a randomly generated latent vector and
    labels as its input and returns an images based on those
    inputs.

    Structure:
        - labels are embedding via Linear neural network layer
        - images and labels are combined and processed through 
          4 convolutional layers each with Batch Normalization 
          and LeakyReLU
        - output layer makes a final convulution before applying
          a TanH function to map the resulting balues into a 
          (-1, 1) range
    """

    def __init__(self, image_size, num_channels, num_labels, latent_vector_size):
        super(CGAN_GeneratorNet, self).__init__()

        # Image Variables
        n_channels = num_channels

        # Label Variables
        n_labels = num_labels
        embedding_size = 50

        # Misc Variables
        z_dim = latent_vector_size
        n_out = image_size

        self.input_size = latent_vector_size + embedding_size

        self.label_embedding = nn.Sequential(
            nn.Linear(n_labels, embedding_size),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Naming arguments for this layer as a guide for other layers
        self.layer_0 = nn.Sequential(
            nn.ConvTranspose2d(z_dim + embedding_size, n_out * 8,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False),
            nn.BatchNorm2d(n_out * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer_1 = nn.Sequential(
            nn.ConvTranspose2d(n_out * 8, n_out * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_out * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer_2 = nn.Sequential(
            nn.ConvTranspose2d(n_out * 4, n_out * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_out * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer_3 = nn.Sequential(
            nn.ConvTranspose2d(n_out * 2, n_out, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_out),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.out = nn.Sequential(
            nn.ConvTranspose2d(n_out, n_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )


    def forward(self, images, labels):
        batch_size = images.size(0)

        x = images.view(batch_size, images.size(1) * images.size(2) * images.size(3))
        y = self.label_embedding(labels)
        x = torch.cat([x, y], 1)
        x = x.view(batch_size, self.input_size, 1, 1)

        x = self.layer_0(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        return self.out(x)














