import torch
from torch import nn

class CGAN_DiscriminatorNet(nn.Module):

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
            
