import pandas as pd
import re
import os
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

    def __init__(self, test_run):
        self.dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../lesion_images')
        #self.dst_dir= os.path.join(self.dir_path, 'all_images_processed')
        self.dst_dir= os.path.join(self.dir_path, 'handpicked_lesion_images_processed_for_gan')
        print("Image destination directory: " + self.dst_dir)
        #self.all_images_path = os.path.join(self.dir_path, 'all_images_cropped_src')
        self.all_images_path = os.path.join(self.dir_path, 'handpicked_lesion_images')
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

        self.test_run = test_run

    def process_all_images(self):
        for filename in self.filenames:
            self.process_image(filename)

    def process_image(self, filename):
        print("Processing: " + filename)

        # First split into list by comma
        # [lname, fname, teethnum, desc1, ..., descn, canalnum, month, day, year, sequencenum]
        params = filename.split(',')

        if re.search("jpg", params[-1]) and len(params) >= 6:
            # We only want to process our images, not files lie .DS_STORE for eg.
            # There are some images with not all the data, so ignore anything less tha 6 params

            # First and Second value will always be names, we can drop those
            # [teethnum, desc1, ..., descn, canalnum, month, day, year, sequencenum]
            params = params[2:]
            print('Current Params: ' + str(params))

            # Extract ID and and lesion binary from last param
            params[-1] = params[-1].split('.jpg')[0] # Remove ".jpg" from the last param
            last_param = params[-1].split('-')
            image_id = last_param[1] # Extract Image Id
            lesion = int(last_param[2]) # Extract lesion binary value
            
            # Extract sequence number
            # Save and drop from list
            # [teethnum, desc1, ..., descn, canalnum, month, day, year]
            sequenceNum = last_param[0]
            print('Sequence Number: ' + sequenceNum)
            params = params[:-1]

            teethNumbers = params[0]
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

                new_file_path = os.path.join(self.dst_dir, new_filename)

                print("saving lesion image: " + filename)
                print("as: " + new_filename)
                print("---")

                image_to_save.save(new_file_path, 'JPEG')


image_processor = ImageProcessor(test_run=False)
image_processor.process_all_images()

dataFrame = pd.DataFrame(data=image_processor.data)
dataFrame.to_csv("./lesion-csv.csv")

