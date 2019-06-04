import torch
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np

from skimage import io

class LesionDataset(Dataset):
    """
    Teeth Lesion Dataset.
    """

    def __init__(self, data_frame, root_dir, transform=None):
        """    
        Args:
            data_frame (): 
            root_dir (string): Directory containing lesion images.
            transform (calable, optional): Optional transform to be applied to images
        """
        self.lesions_df = data_frame
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.lesions_df)

    def __getitem__(self, idx):
        print(" - getting image")
        image_id = self.lesions_df.iloc[idx].imageId
        image = io.imread(self.root_dir + image_id + '.jpg')
        
        print(" - image got")

        # Transforms like resizign should be performed before 
        # importing images into the dataset

        print(" - transforming image for pytorch")
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        print(" - done")
        print(" - normalizing")
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        #image = normalize(image.float())
        print(" - done")

        print(" - loading label")
        label = self.lesions_df.iloc[idx].lesion
        label = label.astype('float')
        label = torch.from_numpy(np.array([label]))
        print(" - done")

        return image, label