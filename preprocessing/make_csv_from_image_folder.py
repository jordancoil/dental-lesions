import os

import argparse

import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create Training CSV using a folder of images.')
    # Right now this only supports building a CSV of 1 class (0 or 1)
    parser.add_argument('--lesion-type', type=int, nargs='?')
    parser.add_argument('--img-dir', type=str, nargs='?')
    parser.add_argument('--csv-name', type=str, nargs='?')

    options = parser.parse_args()

    if options.lesion_type and options.img_dir:
        img_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'lesion_images', options.img_dir)
        csv_target_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data_csvs')

        data = {'imageId': [], 'lesion':[]}

        filenames = os.listdir(img_dir)

        for imagename in filenames:
            data['imageId'].append(imagename.split('.jpg')[0])
            data['lesion'].append(options.lesion_type)

        df = pd.DataFrame(data=data)
        df.to_csv(csv_target_dir + '/' + options.csv_name)
