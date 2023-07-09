import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math

from sklearn.model_selection import train_test_split

from scipy.ndimage import convolve, binary_fill_holes
from scipy.stats import mode

from skimage import io, exposure
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.transform import rotate, hough_line, hough_line_peaks, resize
from skimage.filters import threshold_otsu, sobel, gaussian, threshold_local



def convert_to_grayscale(filepath):
    filename = 'Dataset/' + filepath
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


# CONTRAST STRETCHING
def contrast_stretch(image):
    min_val = image.min()
    max_val = image.max()
    stretched = exposure.rescale_intensity(image, in_range=(min_val, max_val))
    return stretched

# HISTOGRAM EQUALIZATION
def clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(contrast_stretch(image)) 
    return equalized

# BLURRING AND SHARPENING
# Blurring the image
def blur_image(image):
    blur_img = gaussian(clahe(image), sigma = 3)
    return blur_img

# Sharpening image
def sharpen_image(image):
    blurred_image = blur_image(image)
    sharp_img = image + (image - blurred_image)
    return sharp_img

# NORMALIZATION
def normalize_img(image):
    normalized = (sharpen_image(image) - np.min(sharpen_image(image)))/(np.max(sharpen_image(image)) - np.min(sharpen_image(image)))   
    return normalized



#AUGMENTATION FUNCTIONS

#specified rotation
def image_rotation(image, angle_of_rotation):
    
    #rotate the image based on the rotation angle provided
    rotate_img = rotate(image, angle_of_rotation, resize = True)
    
    #resize the image if the size changed
    if rotate_img.shape != image.shape:
        rotated_img = resize(rotate_img, image.shape)
    else:
        rotated_img = rotate_img
     
    #return the final image
    return rotated_img


#random rotation function
def random_rotation(image):
    
    #randomly determine the rotation angle
    random_rotation = np.random.uniform(low = 0.0, high = 360.0)
    
    #rotate the image
    rotate_img = rotate(image, random_rotation, resize = True)
    
    #resize the image if the size changed
    if rotate_img.shape != image.shape:
        rotated_img = resize(rotate_img, image.shape)
    else:
        rotated_img = rotate_img
        
    #return the final image
    return rotated_img


#random flipping function
def random_flip(image):
    
    #generate a method to determine how the image should be flipped
    random_flip_axis = np.random.randint(low = 1, high = 4, dtype = int)
    
    #flip horizontal
    if random_flip_axis == 1:
        flipped_image = image[:, ::-1]
     
    #flip vertical
    elif random_flip_axis == 2:
        flipped_image = image[::-1, :]
    
    #No flip    
    elif random_flip_axis == 3:
        flipped_image = image
    
    #return the flipped image
    return flipped_image
 

#gamma correction - adjusts brightness of image
def gamma_correction(image):
    random_gamma = np.random.uniform(low = 0, high = 1)
    
    gamma_adjusted_image = exposure.adjust_gamma(image, 2)
    return gamma_adjusted_image


#logarithmic correction - replaces pixels with their log value
def log_correction(image):
    log_adjusted_image =  exposure.adjust_log(image)
    return log_adjusted_image


#data augmentation function
def data_augmentation(image):
    random_correction = np.random.randint(low = 1, high = 4, dtype = int)
    
    rotated_image = random_rotation(image)
    flipped_image = random_flip(rotated_image)
    
    #random correction of the image
    if random_correction == 1:
        augmentated_image = gamma_correction(flipped_image)
        
    elif random_correction == 2:
        augmentated_image = log_correction(flipped_image)
    
    elif random_correction == 3:
        augmentated_image = flipped_image
    
    return augmentated_image


def load_images_and_augment():
    path = pd.read_csv('Dataset/cards.csv')
    path_df = pd.DataFrame(path)
    path_df.columns = [c.replace(' ', '_') for c in path_df.columns]
    path_df['suit'] = path_df['labels'].str.split().str[-1]
    path_df = path_df[~path_df['suit'].str.contains('joker', case=False)]
    
    all_images = []
    all_labels = []

    normalized_images = []
    normalized_images_labels = []

    for index, row in path_df.iterrows():
        filepath = row['filepaths']
        label = row['suit']
        try:
            image = convert_to_grayscale(filepath)
            if image is not None and filepath.endswith('.jpg'):
                # Perform any necessary preprocessing on the image
                normalized_images.append(normalize_img(image))
                normalized_images_labels.append(label)
            else:
                continue
        except Exception as e:
                continue


    # append augmented images
    for i in range(len(normalized_images)):    
        all_images.append(normalized_images[i])
        all_labels.append(normalized_images_labels[i])


        augmented_image = data_augmentation(normalized_images[i])
        augmented_image_label = normalized_images_labels[i]

        all_images.append(augmented_image)
        all_labels.append(augmented_image_label)
    return all_images,all_labels

def train_test_validation_split(all_images,all_labels):
    # Splitting into train, test, and validation sets (80% - 10% - 10% split)
    X_train, X_test_val, y_train, y_test_val = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


