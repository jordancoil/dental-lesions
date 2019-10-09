"""
Current Problem:
    Lots of images, but the data pertaining to that image is in the file name.
    Need a better way of organizing everything. This tool will hopefully solve
    that in the following ways:
        - Allow visual manual cropping of images so that:
            - we can select the appropriate regions in the image
            - get all the images into a consistent format
        - Allow manual labelling of the data, and input into a base CSV
"""

"""
1. Develop a tool for manual visual cropping of images

Steps:
    - Given a folder of images, open the images in sequence
    - A crop is selected for that image via a visual interface
    - that image is saved with the crop into a specified folder.
"""

import cv2
import os

def tests():
    test_image = cv2.imread("tests/test_images/test1.jpg")
    cv2.rectangle(test_image, (10, 100), (50, 500), (0, 255, 0), 2)
    cv2.imshow("image", test_image)
    cv2.waitKey(0)

    crop_image = test_image.copy()[100:500, 10:50]
    cv2.imshow("image", crop_image)
    cv2.waitKey(0)
    # tests for crop_image()
    #    assert 

    print("Tests Passed")

def crop_image(image, crop_dim):
    """
    Image [(Number, Number), (Number, Number)] -> Image
    
    takes an image and a coordinates as input and returns the image bounded 
    inside those coordinates
    """
    return image


if __name__ == "__main__":
    tests()
