from collections import defaultdict
import os
from PIL import Image
import numpy as np
from os import path
import pandas as pd
import cv2

DATA_DIR = "tdata"

# function to count number of files per folder
def files_per_folder(folder):
    freq = defaultdict(int)
    for root, dirs, files in os.walk(folder):
        for filename in files:
            filepath = path.join(root, filename)
            label = root.split("/")[-1]
            freq[label] += 1
    print("Total __ images")
    tot = 0
    for k,v in freq.items():
        print("Number of images in folder: ", k, " are: ", v)
        tot += v
    print("Total no. of images: ", tot)
    return tot

# iterate through images in the folders return labels & filepaths
def iter_images():
    for root, dirs, files in os.walk(DATA_DIR):
        label_count=0
        for filename in files:
            filepath = path.join(root, filename)
            label = root.split("/")[-1]
            if label == 'pos':
                label = 1
            elif label=='neg':
                label = 0
            yield label, filepath

# convert grayscale image to array

counter = 0

def image_to_array(img, counter):
    image = cv2.imread(img, 1); 
    image = cv2.resize(image, (28,28))
    cv2.imwrite('/home/jite/Downloads/video/video'+str(counter)+'.png',image)
    return np.array(image, dtype=np.uint8)

total_images = files_per_folder(DATA_DIR)
image_size = 2352 # 28 x 28 x 3
X = np.zeros((total_images, image_size))
label1 = np.zeros((total_images),dtype=str)
label_str = []    
    


if __name__ == '__main__':
    # iterate through respective labels & filepaths
    for i, (label, imgpath) in enumerate(iter_images()):
        # convert image stored in the filpath to array
        arr = image_to_array(imgpath, i)
        # flatten the image array
        X[i]= arr.flatten()
        # store labels for each image
        label_str.append(label)
    labels = np.array(label_str, dtype=str)

# reshape to ensure desired shape of X & labels
X = X.reshape(total_images, -1)
labels = labels.reshape(total_images, -1)

# convert to dataframe in the desired format i.e. same as MNIST.csv
X_final = pd.DataFrame(np.concatenate((labels, X), axis=1))
X_final = X_final.sample(frac=1)

# create and store .csv file for usage in HW_3 upgrade
X_final.to_csv('football_group.csv', index=False)


