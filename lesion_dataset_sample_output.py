import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import skimage.transform as sktrans

from lesion_dataset import LesionDataset

print("Loading sample image from PyTorch dataset...")
print("Reading csv...")
df = pd.read_csv("./train.csv")
df = df.drop(["Unnamed: 0", "Unnamed: 0.1", "teethNumbers", "description", "numberOfCanals", "date", "sequenceNumber"], axis=1)
print("csv read!")

print("loading csv into dataset...")
dataset = LesionDataset(df, "./lesion_images/all_images_processed/")
print("dataset loaded! length: ", len(dataset))

print("displaying image...")
image, label = dataset[0]
print("image and label loaded...")
print("Image Size: ", image.size())
print("Label Size: ", label.size())

image = image.numpy().transpose((1, 2, 0)) # Transposing for pyplot display
image = sktrans.resize(image, (224, 224))
image = np.clip(image, 0, 1)
plt.imshow(image)
plt.show()
