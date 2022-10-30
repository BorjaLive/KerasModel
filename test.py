
from datasetloader import load_dataset
from tensorflow.keras import datasets, layers, models, preprocessing, activations, Input, utils
import numpy as np
from model import zesNet

import pandas as pd
import os
from PIL import ImageFile, Image, ImageDraw
import tensorflow as tf
import random
import math
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.transforms import functional as F


def display_image(image, title, points_pred, points_gt, factor):
    #factor is used to convert the coordinates from between [0,1] to desired image coordinates
    
    plt.imshow(image)
    plt.title(title)
    
    #plots predicted coordinates
    plt.scatter([points_pred[0]*factor*4,points_pred[2]*factor*4],[points_pred[1]*factor*3,points_pred[3]*factor*3], c = 'r')
    #plots ground truth coordinates
    plt.scatter([points_gt[0]*factor*4,points_gt[2]*factor*4],[points_gt[1]*factor*3,points_gt[3]*factor*3], c = 'b')
    plt.show()

img_dir = './Annotations/PTL_Dataset_876x657'
csv_file = './Annotations/training_file.csv'
labels = pd.read_csv(csv_file)

for i in range(1):#len(labels)
    index = 883
    print("index: ", index)

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img_name = os.path.join(img_dir, labels.iloc[index, 0]) #gets image name in csv file
    image = Image.open(img_name)

    light_mode = labels.iloc[index, 1] #mode of the traffic light
    block = labels.iloc[index,6] #label of blocked or unblocked
    points = labels.iloc[index, 2:6] #midline coordinates

    points = [points[0]/4032, points[1]/3024, points[2]/4032, points[3]/3024] #normalize coordinate values to be between [0,1]

    #display_image(image, 'Original Image', points, points, 219)


    #random crop
    cp = [points[0]*876, (1-points[1])*657, 876*points[2], (1-points[3])*657] #convert points to cartesian coordinates
    #shifts to determine what region to crop
    shiftx = random.randint(0, 108) 
    shifty = random.randint(0, 81)

    m = (cp[1]-cp[3])/(cp[0]-cp[2]) #slope
    
    if math.isinf(m): m = 10000000000000000
    
    b = cp[1] - m*cp[0] #y-intercept
    print(m, b)
    
    if(shiftx > cp[0]): 
        cp[0] = shiftx
        cp[1] = (cp[0]*m + b)
    elif((768+shiftx) < cp[0]):
        cp[0] = (768+shiftx)
        cp[1] = (cp[0]*m + b)
    if(shiftx > cp[2]): 
        cp[2] = shiftx
        cp[3] = (cp[2]*m + b)
    elif((768+shiftx) < cp[2]):
        cp[2] = (768+shiftx)
        cp[3] = (cp[2]*m + b)
    if(657-shifty < cp[1]): 
        cp[1] = 657-shifty
        cp[0] = (cp[1]-b)/m if (cp[1]-b)/m>0 else 0
    #       elif((657-576-shifty) > cp[1]):
    #           cp[0] = (657-576-shifty-b)/m
    #           cp[1] = 0
    #           cp[2] = (657-576-shifty-b)/m
    #           cp[3] = 0
    if(657-576-shifty > cp[3]): 
        cp[3] = 657-576-shifty
        cp[2] = (cp[3]-b)/m
    #       elif((657-shifty) < cp[3]):
    #           cp[3] = 657-shifty
    #           cp[2] = (657-shifty-b)/m
    #           cp[1] = 657-shifty
    #           cp[0] = (657-shifty-b)/m

    #converting the coordinates from a 876x657 image to a 768x576 image
    cp[0] -= shiftx
    cp[1] -= (657-576-shifty)
    cp[2] -= shiftx
    cp[3] -= (657-576-shifty)

    #converting the cartesian coordinates back to image coordinates
    points = [cp[0]/768, 1-cp[1]/576, cp[2]/768, 1-cp[3]/576]

    image = F.crop(image, shifty, shiftx, 576, 768)

    display_image(image, 'Transformed Image', points, points, 192)



"""
model = zesNet()
#model.summary()

#utils.plot_model(model)


train_file_dir = './Annotations/training_file.csv'
train_img_dir = './Annotations/PTL_Dataset_876x657'

x, y, z, w = load_dataset(train_file_dir, train_img_dir, total = 1)

print(x[0].shape)
print(y)
print(z)
print(w)
"""
