import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import copy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def convert_to_grayscale(filepath):
    filename = 'Dataset/' + filepath
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

#SIFT based feature detection
#we use SIFT because it is scale and rotation invariant

#move function to a utils file
def sift_featue_detector(image, n_features):
    #initiate the SIFT detector, we look for the 15 best features
    sift = cv2.SIFT_create(nfeatures = n_features)

    #find the key feature points with SIFT
    key_features = sift.detect(image, None)

    #draw key features on the image
    display_key_features = cv2.drawKeypoints(image, key_features, 0, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #display the image with drawn key features with the orientation
    return display_key_features, key_features, sift

def sift_feature_descriptor(image, n_features):
    sift = cv2.SIFT_create()
    keyfeatures = sift_featue_detector(image, n_features)[1]
#     sift = cv.SIFT_create()
    key_features, key_descriptors = sift.compute(image, keyfeatures)
    
    return key_features, key_descriptors

#use the keypoint coordinates to get the relevant pixels

def key_point_pixels(image, n_features):
    image_pixels = []
    for coordinates in sift_featue_detector(image, n_features)[1]:
        x, y = coordinates.pt
        x_coords, y_coords = int(x), int(y)
        key_point_pixel = image[x_coords, y_coords]
        image_pixels.append(key_point_pixel)
    v_len = len(image_pixels)
    
    return image_pixels, v_len

def pyramid_image(template):
    template_image = template
    lower_res_template = cv2.pyrDown(template_image)
    lower_res_template_2 = cv2.pyrDown(lower_res_template)
    lower_res_template_3 = cv2.pyrDown(lower_res_template_2)

    return template_image, lower_res_template, lower_res_template_2, lower_res_template_3

def match_template(template_images, image, method):
    image = image
    original_image = copy.copy(image)
    
    match_title = []
    for template in template_images:
        width, height = template.shape
        match_results = cv2.matchTemplate(image, template, method)
        threshold = 0.9

        location = np.where( match_results >= threshold)

        list_locations = list(zip(*location[::-1]))

        if len(list_locations) > 0:
            title = 'MATCH FOUND'
            print(len(list_locations))
            for pt in zip(*location[::-1]):
                cv2.rectangle(image, pt, (pt[0] + width, pt[1] + height), (0,255,0), 1)
            match_title.append(title)

        else:
            title = 'NO MATCH FOUND'
            match_title.append(title)
    
    if 'MATCH FOUND' in match_title:
        title_results = 'MATCH FOUND'
        
    else:
        title_results = 'NO MATCH FOUND'
        
    return image, original_image, template_images[0], title_results

def strongest_features(data_frame, image_paths, n_features):
    image_kp = []
    image_labels = []
    kp_len = []
    for i in range(len(image_paths)):

        image  = convert_to_grayscale(image_paths[i])
        image = cv2.normalize(image,  None, 0, 255, cv2.NORM_MINMAX)
        label = data_frame['Suit_Label'][i]
        pixels = key_point_pixels(image, n_features)[0]
        v_lengths = key_point_pixels(image, n_features)[1]
        kp_len.append(v_lengths)
        image_kp.append(pixels)
        image_labels.append(label)
        
    return image_kp, image_labels, kp_len