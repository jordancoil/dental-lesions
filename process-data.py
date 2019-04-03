import pandas as pd
import re
import os
import shutil

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
        self.fileindex = 1
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.dst_dir= os.path.join(self.dir_path, 'lesion-images-with-id')
        print("Image destination directory: " + self.dst_dir)
        self.lesion_image_path = os.path.join(self.dir_path, 'lesion-images')
        self.no_lesion_image_path = os.path.join(self.dir_path, 'no-lesion-images')
        self.lesion_filenames = os.listdir(self.lesion_image_path)
        self.no_lesion_filenames = os.listdir(self.no_lesion_image_path)
        self.filenames = self.lesion_filenames + self.no_lesion_filenames

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

    def copy_lesion_image(self, old_filename, image_id, is_lesion):
        new_filename = os.path.join(self.dst_dir, image_id)
        if not os.path.isfile(new_filename):
            if is_lesion:
                src_dir = self.lesion_image_path
            else:
                src_dir = self.no_lesion_image_path
            src_file = os.path.join(src_dir, old_filename) 
            if not self.test_run:
                copy_filename = os.path.join(self.dst_dir, old_filename)
                print("Copying " + src_file)
                print("to: " + copy_filename)
                shutil.copy(src_file, copy_filename)
                dst_file = os.path.join(self.dst_dir, old_filename)
                os.rename(dst_file, image_id)

    def process_lesion_images(self):
        for filename in self.lesion_filenames:
            self.process_image(filename, True)

    def process_no_lesion_images(self):
        for filename in self.no_lesion_filenames:
            self.process_image(filename, False)

    def process_all_images(self):
        self.process_lesion_images()
        self.process_no_lesion_images()

    def process_image(self, filename, hasLesion):
        print("Processing: " + filename)

        # First split into list by comma
        # [lname, fname, teethnum, desc1, ..., descn, canalnum, month, day, year, sequencenum]
        params = filename.split(',')

        if re.search("JPG", params[-1]) and len(params) >= 6:
            # We only want to process our images, not files lie .DS_STORE for eg.
            # There are some images with not all the data, so ignore anything less tha 6 params

            # First and Second value will always be names, we can drop those
            # [teethnum, desc1, ..., descn, canalnum, month, day, year, sequencenum]
            params = params[2:]
            print('Current Params: ' + str(params))

            self.data['teethNumbers'].append(params[0])
            params = params[1:]

            # Last value should always be sequence number
            # Save and drop from list
            # [teethnum, desc1, ..., descn, canalnum, month, day, year]
            sequenceNum = params[-1].split('.JPG')[0]
            print('Sequence Number: ' + sequenceNum)
            self.data['sequenceNumber'].append(sequenceNum)
            params = params[:-1]
        
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
            self.data['date'].append(pd.to_datetime(formattedDate, format='%Y%m%d'))
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

            self.data['numberOfCanals'].append(canal_to_add)
            params = new_params

            # If there are any values left, they will be considered part of the description.
            # We can join them all into a string
            if len(params):
                self.data['description'].append(",".join(params))
            else:
                self.data['description'].append("")

            self.data['lesion'].append(1 if hasLesion else 0)

            lesion_text = "lesion" if hasLesion == True else "nolesion"
            lesion_id = str(self.fileindex) + "-" + lesion_text + ".JPG"
            self.data['imageId'].append(lesion_id)

            print("saving lesion image: " + filename)
            print("as: " + lesion_id)
            print("---")
            self.copy_lesion_image(filename, lesion_id, hasLesion)
            self.fileindex += 1

image_processor = ImageProcessor(test_run=False)
image_processor.process_all_images()

dataFrame = pd.DataFrame(data=image_processor.data)
dataFrame.to_csv("./lesion-csv.csv")

