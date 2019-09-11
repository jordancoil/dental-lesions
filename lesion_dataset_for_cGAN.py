import torch
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np

from skimage import io

class LesionDatasetCGAN(Dataset):
    """
    This Dataset is very similar to the one defined 
    in the file lesion_dataset.py. However this class
    has the added functionality of returning extra
    data about each sample. For instance, related
    tooth number, which theorhetically could be used
    by a conditional GAN to improve synthetic image
    quality.
    """

    def __init__(self, data_frame, root_dir, transform=None):
        """
        Args:
            data_frame (?): the pandas dataframe containing our data.
            root_dir (string): Directory containing lesion images.
            transform (callable, optional): Optional transforms to
                apply to images.
        """
        self.lesions_df = data_frame
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.lesions_df)

    def __getitem__(self, idx):
        image_id = self.lesions_df.iloc[idx].imageId
        image = io.imread(self.root_dir + image_id + '.jpg')

        # Greyscale reshape
        # image = image.reshape((image.shape[0], image.shape[1], 1))

        if self.transform:
            image = self.transform(image)

        to_tensor = transforms.ToTensor()
        image = to_tensor(image)

        # Perform image normalization if needed
        # normalize = transforms.Normalize(mean=[], std=[])

        lesion = self.lesions_df.iloc[idx].lesion
        lesion = lesion.astype('float')

        tooth_num = self.lesions_df.iloc[idx].teethNumbers
        tooth_num = tooth_num.astype('float')

        labels = torch.from_numpy(np.array([lesion, tooth_num]))

        return image, labels
