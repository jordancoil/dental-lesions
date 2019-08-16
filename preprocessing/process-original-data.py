import pandas as pd
import re
import os
import argparse

from datetime import datetime

import shutil
from PIL import Image

# data is stored within the filenames of the images
# the structure is as follows:
# {lastName},{firstName},{teethNumbers},{description},{numberOfCanals},{date},{sequenceNumber}
# {teethNumbers}   - the dental code values for the teeth present in the image
# {description}    - could have multiple values, each with their own comma
# {numberOfCanals} - an optional field, marking the number of canals present in the tooth
# {date}           - the date in M/D/Y format, comma separated.
# {sequenceNumber} - each case may have multiple images, over multiple dates, 
#                    the sequence number is the order of the images taken on the same date
#
# we need to strip out names to ensure confidentiality.
# we could put this into a pandas dataframe, and then we would be able to get information about each field
# for instance, we can verify that we processed the data correctly by checking the types in each column

class ImageProcessor:

    def __init__(self, test_run, process_images, output_folder, input_folder):
        self.dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../lesion_images')
        self.dst_dir= os.path.join(self.dir_path, output_folder)
        print("Image destination directory: " + self.dst_dir)
        #self.all_images_path = os.path.join(self.dir_path, 'all_images_cropped_src')
        self.all_images_path = os.path.join(self.dir_path, input_folder)
        self.filenames = os.listdir(self.all_images_path)

        self.data = {
            'imageId': [], # We also should copy the images to image ids, and encode that as well
            'teethNumbers': [],
            'description': [],
            'numberOfCanals': [],
            'date': [],
            'sequenceNumber': [],
            'lesion': []
        }

        self.tooth_num_map = {
            '0': '0',
            '18': '1',
            '17': '2',
            '16': '3',
            '15': '4',
            '14': '5',
            '13': '6',
            '12': '7',
            '11': '8',
            '21': '9',
            '22': '10',
            '23': '11',
            '24': '12',
            '25': '13',
            '26': '14',
            '27': '15',
            '28': '16',
            '38': '17',
            '37': '18',
            '36': '19',
            '35': '20',
            '34': '21',
            '33': '22',
            '32': '23',
            '31': '24',
            '41': '25',
            '42': '26',
            '43': '27',
            '44': '28',
            '45': '29',
            '46': '30',
            '47': '31',
            '48': '32',
        }

        self.process_images = process_images

        self.test_run = test_run

    def process_all_images(self):
        for file_id, filename in enumerate(self.filenames):
            self.process_image(filename, file_id)

    def process_image(self, filename, file_id):
        print("Processing: " + filename)

        # First split into list by comma
        # [lname, fname, teethnum, desc1, ..., descn, canalnum, month, day, year, sequencenum]
        params = filename.split(',')

        if re.search("(?:jpg|JPG|JPEG)", params[-1]) and len(params) >= 6:
            # We only want to process our images, not files lie .DS_STORE for eg.
            # There are some images with not all the data, so ignore anything less tha 6 params

            # First and Second value will always be names, we can drop those
            # [teethnum, desc1, ..., descn, canalnum, month, day, year, sequencenum]
            params = params[2:]
            print('Current Params: ' + str(params))

            # Extract ID and and lesion binary from last param
            jpeg_term = re.search("(?:jpg|JPG|JPEG)", params[-1])[0]
            params[-1] = params[-1].split('.'+jpeg_term)[0] # Remove ".jpg" from the last param
            last_param = params[-1].split('-')
            image_id = last_param[1] # Extract Image Id
            lesion = int(last_param[2]) # Extract lesion binary value
            
            # Extract sequence number
            # Save and drop from list
            # [teethnum, desc1, ..., descn, canalnum, month, day, year]
            sequenceNum = last_param[0]
            print('Sequence Number: ' + sequenceNum)
            params = params[:-1]

            teethNumbers = self.tooth_num_map[params[0].split('-')[0][:2]]
            params = params[1:]

            # Extract date
            date = params[-3:]
            print("Date: "+ str(date))
            # Format date
            date[2] = '20' + date[2]
            if len(date[0]) == 1:
                date[0] = '0' + date[0]

            #               year      month     day
            formattedDate = date[2] + date[0] + date[1]
            print("formatted date: " + formattedDate)
            params = params[:-3]

            # Use regex to determine if Number of Canals var is present
            # It would be in the format {number}c
            # Sometimes the number of canals is present in the description.
            # So therefore we should loop through the list at this point to find the canal number
            already_matched = False
            new_params = []
            canal_to_add = ""
            for index, param in enumerate(params):
                match = re.search("[0-9][c]", param)
                if match and not already_matched:
                    canal_to_add = param
                else:
                    new_params.append(param)

            params = new_params

            img = Image.open(os.path.join(self.all_images_path, filename))
            rotated = [img, img.rotate(90), img.rotate(180), img.rotate(270)]
            for index, image_to_save in enumerate(rotated):
                image_id = str(file_id) # overwrite old image id with new iterative id
                new_file_id = image_id + "-" + str(index)
                new_filename = new_file_id + ".jpg"

                self.data['imageId'].append(new_file_id)
                self.data['sequenceNumber'].append(sequenceNum)
                self.data['teethNumbers'].append(teethNumbers)
                self.data['date'].append(pd.to_datetime(formattedDate, format='%Y%m%d'))
                self.data['numberOfCanals'].append(canal_to_add)
                # If there are any values left, they will be considered part of the description.
                # We can join them all into a string
                if len(params):
                    self.data['description'].append(",".join(params))
                else:
                    self.data['description'].append("")

                self.data['lesion'].append(lesion)

                if self.process_images:
                    new_file_path = os.path.join(self.dst_dir, new_filename)

                    print("saving lesion image: " + filename)
                    print("as: " + new_filename)
                    print("---")

                    image_to_save.save(new_file_path, 'JPEG')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='process folder of labeled images into csv and images ready for training.')
    parser.add_argument('--process-images', dest='process_images', action='store_true')
    parser.add_argument('--csv-name', type=str, nargs='?')
    parser.add_argument('--output-dir', type=str, nargs='?')
    parser.add_argument('--input-dir', type=str, nargs='?')
    options = parser.parse_args()

    if options.process_images and not options.output_dir:
        print('Please specify an output dir using the argument "--output-dir"')
        exit()
    elif not options.output_dir:
        output_dir = 'placeholder'

    if options.csv_name:
        csv_name = options.csv_name + '.csv'
    else:
        csv_name = str(datetime.now()) + '_lesions.csv'

    if options.input_dir:
        input_dir = options.input_dir
    else:
        input_dir = '1_images-first-set/all_images_cropped_src/'

    process_images = options.process_images
    output_dir = options.output_dir

    image_processor = ImageProcessor(False, process_images, output_dir, input_dir)
    image_processor.process_all_images()

    dataFrame = pd.DataFrame(data=image_processor.data)
    dataFrame.to_csv("../data_csvs/" + csv_name)

