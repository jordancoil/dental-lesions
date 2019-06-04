import os

from skimage import io
import skimage.transform as sktrans

"""
Before loading the images into PyTorch we should perform all preprocessing

Preprocessing steps included in this file:
    1. Resizing to 224, 224
"""

all_images_parent_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..','lesion_images')
target_images_dir = os.path.join(all_images_parent_dir, 'all_images_processed')
new_dir = os.path.join(all_images_parent_dir, 'all_images_processed_2')

filenames = os.listdir(target_images_dir)
progress = 1

for imagename in filenames:
    image = io.imread(target_images_dir + '/' + imagename)
    image = sktrans.resize(image, (224,224))

    new_image_path = os.path.join(new_dir, imagename)

    io.imsave(new_image_path, image)

    if progress % 20 == 0:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("Progress: ", progress / len(filenames) * 100 )

    progress += 1

