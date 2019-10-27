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

import argparse
import cv2
import os

# =========
# Constants

# Global Var(s) used by cropping functions.
rect = [(0,0), (0,0)] # [(start_x, start_y), (end_x, end_y)]
crop_dim = (100, 100)
move_rect = False
curr_img = None
curr_img_copy = None

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
        image_from_function = draw_rectangle(test_image, [(10, 100), (50, 500)])
        assert assert_images_equal(image_with_rectangle, image_from_function) == True
    except Exception as e:
        print("Draw rectangle failed")
        raise e

    # test init_global_crop_dim()
    try:
        test_height, test_width = test_image.shape[:2]
        test_dim = min(test_width, test_height)
        assert (test_dim, test_dim) == init_global_crop_dim(test_image, save=False)
    except exception as e:
        print("Init crop dim failed")
        raise e

    # test check_crop_out_of_bounds()
    try:
        global crop_dim
        start_x, start_y, end_x, end_y = check_crop_out_of_bounds(test_image, 
                int(0.5*crop_dim[0]) - 1, int(0.5*crop_dim[1]) - 1)
        assert start_x == 0
        assert start_y == 0
        assert end_x   == crop_dim[0]
        assert end_y   == crop_dim[1]

        start_x, start_y, end_x, end_y = check_crop_out_of_bounds(test_image, 
                test_image.shape[1] - int(0.5*crop_dim[0]) + 1, 
                test_image.shape[0] - int(0.5*crop_dim[1]) + 1)
        assert start_x == test_image.shape[1] - crop_dim[1]
        assert start_y == test_image.shape[0] - crop_dim[0]
        assert end_x   == test_image.shape[1]
        assert end_y   == test_image.shape[0]
    except Exception as e:
        print("Crop out of bounds check function failed")
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


def draw_rectangle(image, rect):
    """
    Image, [(Number, Number), (Number, Number)] -> Image

    takes an Open CV image and some coordinates and draws a green rectangle around 
    those coordinates in OpenCV
    """
    return cv2.rectangle(image, rect[0],     # Start Point
                                rect[1],     # End Point
                                (0, 255, 0), # Color of Rectangle
                                2)           # Thickness

    
    
def open_image_and_start_crop(image):
    """
    Image -> Image

    takes an Open CV image and displays it using Open CV. Waits for the press
    of the escape key to exit or the s key to save and exit.
    """
    global rect, curr_img, curr_img_copy

    #TODO: move this to the function that opens the directory of images
    #   the idea is that the crop size will be set to a square of the min
    #   of the smallest images width or height 
    init_global_crop_dim(image)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", move_rectangle)
    curr_img = image
    curr_img_copy = curr_img.copy()
    cv2.startWindowThread()
    while True:
        cv2.imshow('image', curr_img)
        # check every 100ms to see if window mannualy closed
        k = cv2.waitKey(100) 
        if k == 27: 
            # Esc key is pressed
            cv2.destroyAllWindows()
            break
        elif k == ord('s'):
            image = crop_image(image, rect)
            break
        if cv2.getWindowProperty("image", cv2.WND_PROP_VISIBLE) < 1:
            # Image is closed manually
            break
    cv2.destroyAllWindows()

    return image


def init_global_crop_dim(image, save=True):
    """
    Image, (Boolean) -> (Number, Number)

    takes an image and sets the global crop width and height to to the min of 
    the image's width or it's height, returns a tuple of (width, heigt)

    Optional save paramter for testing purposes.
    """
    global crop_dim

    height, width = image.shape[:2]
    dim = min(width, height)

    if save:
        crop_dim = (dim, dim)

    return (dim, dim)


def get_crop_coords(image, x, y):
    """
    Image, Number, Number -> [(Number. Number), (Number, Number)]

    Takes an Image and dimensions of a rectangle and returns those dimensions 
    coordinates for a crop function. If the x and y coords would put the 
    rectangle off the screen, set the edge of the rectangle to nearest edge.

    """
    start_x, start_y, end_x, end_y = check_crop_out_of_bounds(image, x, y)
    
    return [(start_x, start_y), (end_x, end_y)]


def check_crop_out_of_bounds(image, x, y):
    """
    Image, Number, Number, (Number, Number) ->
        Number, Number, Number, Number
    """
    # TODO: check if crop coords are out of bounds
    global crop_dim

    img_h, img_w = image.shape[:2]
    
    crop_w, crop_h = crop_dim
    start_x = x - int(0.5*crop_w)
    start_y = y - int(0.5*crop_h)
    end_x   = x + int(0.5*crop_w)
    end_y   = y + int(0.5*crop_h)

    if start_x < 0:
        start_x = 0
        end_x = crop_w
    if start_y < 0:
        start_y = 0
        end_y = crop_h
    if end_x > img_w:
        end_x = img_w
        start_x = img_w - crop_w
    if end_y > img_h:
        end_y = img_h
        start_y = img_h - crop_h

    return start_x, start_y, end_x, end_y


def store_rect(new_rect):
    """
    [(Number, Number), (Number, Number)] -> 
        [(Number, Number), (Number, Number)] 

    Takes a rect (coordinates) and stores it in the global rect variable to be 
    used by open functions.
    """
    global rect
    rect = new_rect
    return rect

def move_rectangle(event, x, y, flags, params):
    """
    Int, Int, Int, Int, (UserData) -> null

    An Open CV mouse event function, drawing an rectangle indicating an area
    to be cropped. The ending position of a mouse after a click should be the 
    center of a new rectangle.    
    """
    global move_rect, crop_size, curr_img, curr_img_copy
    # TODO: use crop_size

    if event == cv2.EVENT_LBUTTONDOWN:
        move_rect = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if move_rect:
            # re-initialize the image, to remove old rectangle
            curr_img = curr_img_copy.copy()
            rect = store_rect(get_crop_coords(curr_img, x, y))
            curr_img = draw_rectangle(curr_img, rect)
    elif event == cv2.EVENT_LBUTTONUP:
        move_rect = False
        # re-initialize the image, to remove old rectangle
        curr_img = curr_img_copy.copy()
        rect = store_rect(get_crop_coords(curr_img, x, y))
        curr_img = draw_rectangle(curr_img, rect)


def open_image_directory(dir_path):
    """
    String -> Array[Image]

    Given a path string, check all files in directory are images and return 
    the images contained in the directory as a list
    """
    try:
        check_all_images(dir_path)
        return os.list_dir(dir_path)
    except Exception as e
        print("Input directory contains non-image files")
        raise e


def check_all_images(dir_path):
    """
    String -> Boolean

    checks that all the files in a given directory are image files
    """
    # TODO
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manually Process Images")
    parser.add_argument("--input_dir", type=str, nargs="?")
    parser.add_argument("--output_dir", type=str, nargs="?")
    options = parser.parse_args()
    argument_errors = 0

    if not options.input_dir:
        print("please specify an input directory using '--input_dir'")
        argument_errors += 1
    if not options.output_dir:
        print("please specify an output directory using '--output_dir'")
        argument_errors += 1
    if argument_errors > 0:
        exit()

    tests()
    test_image = cv2.imread(test_image_path)
    new_image = open_image_and_start_crop(test_image)
