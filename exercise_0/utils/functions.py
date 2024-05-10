import os
import numpy as np
from typing import List, Tuple
import cv2

# Define custom types for clarity
t_image_list = List[np.array]  # List of numpy arrays representing images
t_str_list = List[str]          # List of strings representing names or filenames
t_image_triplet = Tuple[np.array, np.array, np.array]  # Tuple of three numpy arrays representing images


def show_images(images: t_image_list, names: t_str_list) -> None:
    """
    Display images using OpenCV's imshow function.
    
    Args:
        images (t_image_list): List of images as numpy arrays.
        names (t_str_list): List of names corresponding to each image.
    """
    for img, name in zip(images, names):
        cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

def save_images(images: t_image_list, filenames: t_str_list, resource_folder: str = "resources") -> None:
    """
    Save images to specified filenames in the given resource folder.
    
    Args:
        images (t_image_list): List of images as numpy arrays.
        filenames (t_str_list): List of filenames corresponding to each image.
        resource_folder (str): Folder where images will be saved. Default is "resources".
    """
    for image, filename in zip(images, filenames):
        output_path = os.path.join(resource_folder, filename)
        cv2.imwrite(output_path, image)


def scale_down(image: np.array) -> np.array:
    """
    Scale down the input image by a factor of 0.5 in both dimensions.
    
    Args:
        image (np.array): Input image as a numpy array.
        
    Returns:
        np.array: Scaled down image as a numpy array.
    """
    scaled_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    return scaled_image


def separate_channels(colored_image: np.array) -> t_image_triplet:
    """
    Separate the color channels (Blue, Green, Red) of a colored image.
    
    Args:
        colored_image (np.array): Input colored image as a numpy array.
        
    Returns:
        t_image_triplet: Tuple containing three numpy arrays representing Blue, Green, and Red channels.
    """
    # Split the Blue, Green, and Red color channels
    blue, green, red = cv2.split(colored_image)

    # Define channel having all zeros
    zeros = np.zeros(blue.shape, np.uint8)

    # Merge zeros to make BGR image for each channel
    blueBGR = cv2.merge([blue, zeros, zeros])
    greenBGR = cv2.merge([zeros, green, zeros])
    redBGR = cv2.merge([zeros, zeros, red])

    return blueBGR, greenBGR, redBGR
