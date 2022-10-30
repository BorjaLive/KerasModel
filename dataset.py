import pandas as pd
import os
import numpy as np
from PIL import ImageFile, Image, ImageDraw
import tensorflow as tf
import random
import math

from torchvision import transforms
from torchvision.transforms import functional as F

class TrafficLightDataset(tf.keras.utils.Sequence):

    def __init__(self, csv_file, img_dir, batch_size):
        self.labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.batch_size = batch_size

    def on_epoch_end(self):
        pass

    def __getdata__(self, index):
        
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img_name = os.path.join(self.img_dir, self.labels.iloc[index, 0]) #gets image name in csv file
        image = Image.open(img_name)

        light_mode = self.labels.iloc[index, 1] #mode of the traffic light
        block = self.labels.iloc[index,6] #label of blocked or unblocked
        points = self.labels.iloc[index, 2:6] #midline coordinates
        points = [points[0]/4032, points[1]/3024, points[2]/4032, points[3]/3024] #normalize coordinate values to be between [0,1]

        #random horizontal flip with 50% probability
        num = random.random()
        if num >= 0.5:
            image = F.hflip(image)
            #flip x coordinates when entire image is flipped
            points[0] = 1 - points[0] 
            points[2] = 1 - points[2]

        #random crop
        cp = [points[0]*876, (1-points[1])*657, 876*points[2], (1-points[3])*657] #convert points to cartesian coordinates
        #shifts to determine what region to crop
        shiftx = random.randint(0, 108) 
        shifty = random.randint(0, 81)

        m = (cp[1]-cp[3])/(cp[0]-cp[2]) #slope
        
        if math.isinf(m): m = 10000000000000000 #if slope is infinite, set it to a very large number

        b = cp[1] - m*cp[0] #y-intercept

        #changing the coordinates based on the new cropped area
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
        #image = image.crop(shifty, shiftx, 576, 768)
        transform = transforms.Compose([transforms.ColorJitter(0.05,0.05,0.05,0.01)])
        image = transform(image)

        #normalize image
        #image = transforms.functional.to_tensor(image)
        #image = transforms.functional.normalize(image, mean = [120.56737612047593, 119.16664454573734, 113.84554638827127], std=[66.32028460114392, 65.09469952002551, 65.67726614496246])
        
        image = np.transpose(image, (1, 0, 2))
        #points = torch.FloatTensor(points)

        #final_label = {'image': image, 'mode':light_mode, 'points': points, 'block': block}
        #return final_label
        return image, light_mode, points, block

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        batch_x = []
        batch_y = []
        batch_z = []
        for i in range(start, end):
            x, y, z, w = self.__getdata__(i)
            batch_x.append(x)
            batch_y.append(y)
            batch_z.append(z)
        return np.array(batch_x), [np.array(batch_y), np.array(batch_z)]

    def __len__(self):
        return math.ceil(len(self.labels) / self.batch_size)
