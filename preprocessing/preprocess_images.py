import os

import argparse

from skimage import io
from skimage.color import rgb2gray
import skimage.transform as sktrans

"""
Before loading the images into PyTorch we should perform all preprocessing

Preprocessing steps included in this file:
    1. Resizing to 224, 224 or target size from args
    2. Renaming
    3. Input and Output Folders
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess Lesion Images')
    parser.add_argument('--rename', dest='rename', action='store_true')
    parser.add_argument('--input-dir', type=str, nargs='?')
    parser.add_argument('--output-dir', type=str, nargs='?')
    parser.add_argument('--target-size', type=int, nargs='?')
    parser.add_argument('--black-white', dest='black_white', action='store_true')
    options = parser.parse_args()

    if options.input_dir and options.output_dir:
        all_images_parent_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..','lesion_images')
        target_images_dir = os.path.join(all_images_parent_dir, options.input_dir)
        new_dir = os.path.join(all_images_parent_dir, options.output_dir)

        filenames = os.listdir(target_images_dir)
        progress = 1

        for imagename in filenames:
            image = io.imread(target_images_dir + '/' + imagename)
            if options.target_size:
                target_size = (options.target_size, options.target_size)
            else:
                target_size = (224, 224)
            image = sktrans.resize(image, target_size)

            if options.black_white:
                image = rgb2gray(image)

            if options.rename:
                """
                Image names are expected to follow the convention:
                    - blahblah,blahbalh,f1-(0 or 1).jpg
                So we will keep the last number.
                """
                imagename_list = imagename.split(',')
                new_name = imagename_list[-1]
                if new_name.startswith('f'):
                    new_list = new_name.split('-')
                    new_list[0] = ''
                    new_name = '-'.join(new_list)
                    new_name = new_name[1:]

            else:
                new_name = imagename

            new_image_path = os.path.join(new_dir, new_name)

            io.imsave(new_image_path, image)

            if progress % 20 == 0:
                os.system('cls' if os.name == 'nt' else 'clear')
                print("Progress: ", progress / len(filenames) * 100 )

            progress += 1
    else:
        print('Please specify an input dir (--input-dir MyDir) and an output dir (--output-dir MyNewDir)')
