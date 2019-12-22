import argparse
from datetime import datetime
import os
import pandas as pd
from PIL import Image
import re


class Preprocess:
    """
    This class contains all methods for preprocessing images.
    """

    def __init__(self):
        self.img_dir_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'lesion_images/')
        self.csv_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'data_csvs/')

        # To be initialized by preprocessing methods.
        self.inp_dir = ""
        self.out_dir = ""

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

    def processFilenames(self, inp_dir, out_dir):
        self._initInputOutputDirectories(inp_dir, out_dir)
        filenames = os.listdir(self.inp_dir)

        self.data = {
            'image_id': [],
            'teeth_numbers': [],
            'description': [],
            'number_of_canals': [],
            'date': [],
            'sequence_number': [],
            'lesion': []
        }

        for file_i, filename in enumerate(filenames):
            self.processFilename(file_i, filename)

        csv_name = str(datetime.now()) + '_lesions.csv'

        dataFrame = pd.DataFrame(data=self.data)
        dataFrame.to_csv(self.csv_dir + csv_name)

    def processFilename(self, file_i, filename):
        # TODO: Create class that will log outputs of preprocessing scripts

        # First split into list by comma
        # [lname, fname, teethnum, desc1, ..., descn, canalnum, month, day,
        #  year, sequencenum]
        params = filename.split(',')

        if re.search("(?:jpg|JPG|JPEG)", params[-1]) and len(params) >= 6:
            # We only want to process our images, ie. not files like .DS_STORE
            # There are some images with not all the data, so ignore anything
            # less tha 6 params

            # First and Second value will always be names, we can drop those
            # [teethnum, desc1, ..., descn, canalnum, month, day, year,
            #  sequencenum]
            params = params[2:]

            print(params)

            # Extract ID and and lesion binary from last param
            jpeg_term = re.search("(?:jpg|JPG|JPEG)", params[-1])[0]
            # Remove ".jpg" from the last param
            params[-1] = params[-1].split('.'+jpeg_term)[0]
            last_param = params[-1].split('-')
            if len(last_param) < 2:
                # Image has not yet been classified
                return

            lesion = int(last_param[-1]) # Binary value.

            sequence_num = last_param[0]
            # [teethnum, desc1, ..., descn, canalnum, month, day, year]
            params = params[:-1]

            teeth_numbers = self.tooth_num_map[params[0].split('-')[0][:2]]
            # [desc1, ..., descn, canalnum, month, day, year]
            params = params[1:]

            # Extract and format date
            date = params[-3:]
            date[2] = '20' + date[2]
            if len(date[0]) == 1:
                date[0] = '0' + date[0]

            #                year      month     day
            formatted_date = date[2] + date[0] + date[1]
            # [desc1, ..., descn, canalnum]
            params = params[:-3]

            # Use regex to determine if Number of Canals var is present
            # It would be in the format {number}c
            # Sometimes the number of canals is present in the description. So
            # therefore we should loop through the list at this point to find
            # the canal number.
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

            # Augment data by saving 4 90 degree rotations
            img = Image.open(os.path.join(self.inp_dir, filename))
            rotated = [img, img.rotate(90), img.rotate(180), img.rotate(270)]
            for index, image_to_save in enumerate(rotated):
                # Overwrite old image id with new iterative id
                image_id = str(file_i)
                new_file_id = image_id + "-" + str(index)
                new_filename = new_file_id + ".jpg"

                self.data['image_id'].append(new_file_id)
                self.data['teeth_numbers'].append(teeth_numbers)
                if len(params):
                    self.data['description'].append(",".join(params))
                else:
                    self.data['description'].append("")
                self.data['number_of_canals'].append(canal_to_add)
                self.data['date'].append(
                    pd.to_datetime(formatted_date, format='%Y%m%d'))
                self.data['sequence_number'].append(sequence_num)

                # If there are any values left, they will be considered part
                # of the description. We can join them all into a string
                self.data['lesion'].append(lesion)

                new_file_path = os.path.join(self.out_dir, new_filename)

                image_to_save.save(new_file_path, 'JPEG')


    def _initInputOutputDirectories(self, inp_dir, out_dir):
        self.inp_dir = os.path.join(self.img_dir_path, inp_dir)
        self.out_dir = os.path.join(self.img_dir_path, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images.')
    parser.add_argument('-o', '--output-dir', type=str, nargs='?')
    parser.add_argument('-i', '--input-dir', type=str, nargs='?')
    parser.add_argument('-p', '--process', type=str, nargs='?')
    options = parser.parse_args()

    if not options.output_dir:
        print('Please specify an output dir using the argument "--output-dir"')
        exit()

    if not options.input_dir:
        print('Please specify an output dir using the argument "--input-dir"')
        exit()

    if not options.process:
        print('Please specify a process using the argument "--process"')
        exit()

    image_processor = Preprocess()

    input_dir = options.input_dir
    output_dir = options.output_dir

    if options.process in ["1", "filenames"]:
        image_processor.processFilenames(input_dir, output_dir)
