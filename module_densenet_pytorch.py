import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


class DenseNetCNN(nn.Module):

    def __init__(self, num_classes):
        super(DenseNetCNN, self).__init__()

        self.densenet = models.densenet161(pretrained=True)

        in_features = self.densenet.classifier.in_features
        out_features = 32 * 53 * 53
        self.densenet.classifier = nn.Linear(in_features, out_features)

        self.fc1 = nn.Linear(out_features, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.out = nn.Sigmoid()

        self.out_features = out_features

    def forward(self, x):
        x = self.densenet(x)
        x = x.view(-1, self.out_features)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(self.fc3(x))

        return x
