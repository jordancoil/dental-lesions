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

# =========
# Constants

# Global Var(s) used by cropping functions.
rect = []

test_image_path = "tests/test_images/test1.jpg"

def tests():
    test_image = cv2.imread(test_image_path)

    # Test assert_images_equal()
    assert assert_images_equal(test_image, test_image) == True

    # Test crop_image()

    duplicate_cropped = test_image.copy()[100:500, 10:50]
    cropped_test_image = crop_image(test_image, [(10, 100), (50, 500)])
    try:
        assert_images_equal(cropped_test_image, duplicate_cropped)
    except Exception as e:
        print("Crop failed")
        raise e

    try:
        image_with_rectangle = cv2.rectangle(test_image, (10, 100),
                (50, 500), (0, 255, 0), 2)
        image_from_function = draw_rectangle(test_image, 10, 100, 50, 500)
        assert assert_images_equal(image_with_rectangle, image_from_function) == True
    except Exception as e:
        print("Draw rectangle failed")
        raise e

    print("Tests Passed")



def assert_images_equal(original, duplicate):
    assert original.shape == duplicate.shape, "The Images do not have the same shape"

    difference = cv2.subtract(original, duplicate)
    b, g, r = cv2.split(difference)
    assert cv2.countNonZero(b) == 0, "The Images are not equal"
    assert cv2.countNonZero(g) == 0, "The Images are not equal"
    assert cv2.countNonZero(r) == 0, "The Images are not equal"

    return True


def crop_image(image, crop_dim):
    """
    Image [(Number, Number), (Number, Number)] -> Image
    
    takes an image and a coordinates [(start_x, start_y), (end_x, end_y)] 
    as input and returns the image bounded inside those coordinates
    """

    return image[crop_dim[0][1]:crop_dim[1][1], crop_dim[0][0]:crop_dim[1][0]]


def draw_rectangle(image, start_x, start_y, end_x, end_y):
    """
    Image, Number, Number, Number, Number -> Image

    takes an Open CV image and some coordinates and draws a green rectangle around 
    those coordinates in OpenCV
    """

    return cv2.rectangle(image, (start_x, start_y), # Start Point
                                    (end_x, end_y),     # End Point
                                    (0, 255, 0),        # Color of Rectangle
                                    2)                  # Thickness

    
    
def open_image_and_start_crop(image):
    """
    Image -> Image

    takes an Open CV image and displays it using Open CV. Waits for the press
    of the escape key to exit or the s key to save and exit.
    """
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", change_crop)

    cv2.imshow('image', image)
    k = cv2.waitKey(0)
    if k == 27: # wait for ESC key to exit
        cv2.destroyAllWindows()
    elif k == ord('s'):
        crop_dim = get_crop_coordinates(image, rect) # global rect
        cropped_image = crop_image(image, crop_dim)


def get_crop_coordinates(image, crop_rect):
    """
    Image, [(Number, Number), (Number, Number)] ->
        [(Number. Number), (Number, Number)]

    Takes an Image and dimensions of a rectangle and returns those dimensions 
    coordinates for a crop function
    """
    # TODO
    return [(0, 0) (0, 0)]


def change_crop(event, x, y, flags, params):
    """
    Int, Int, Int, Int, (UserData) -> Boolean

    An Open CV mouse event function, where the ending position of the mouse
    should be the center of the new crop. If the end position of the mouse
    would put the crop off the screen, set crop to nearest edge. Returns
    True if everything works.
    """
    # TODO
    global rect

    return True


if __name__ == "__main__":

    tests()
    test_image = cv2.imread(test_image_path)
    open_image_and_start_crop(test_image)
