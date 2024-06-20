# -*- coding: utf-8 -*-
"""
Created on Wed May 20 20:36:13 2020

@author: C-Non
"""

import os
from PIL import Image    
import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt
import cv2 
import pandas as pd
from sklearn import tree

def test2(imgpathget):
    path_of_images = r""+str(imgpathget)
    list_of_images = os.listdir(path_of_images)
    print(path_of_images)
    print(list_of_images)
    for image in list_of_images:
        img = cv2.imread(os.path.join(path_of_images, image))
        print(img)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(rgb)
        img_find = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret,thresh1 = cv2.threshold(img_find,3,255,cv2.THRESH_BINARY)
        
        connectivity = 4
        output = cv2.connectedComponentsWithStats(thresh1, connectivity, cv2.CV_32S)       
        
        left = output[2][1,0]
        top = output[2][1,1]
        right = output[2][1,0]+output[2][1,2]
        bottom = output[2][1,1]+output[2][1,3]
        
        crop_img = img[top-30:bottom+30,left-30:right+30]    
        
        width = crop_img.shape[1]
        height = crop_img.shape[0]
        
        scale_percent = 60
        width_resize = int(width * scale_percent / 100)
        height_resize = int(height * scale_percent / 100)
        dim = (width_resize, height_resize)
        
        resized = cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA)
        rgb_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        
        
        green_lower_range = np.array([25,0,0])
        green_upper_range = np.array([60,255,255])
        mask_g = cv2.inRange(rgb_resized, green_lower_range, green_upper_range)
        mask_g = mask_g.astype('bool')
        g_region = rgb_resized * np.dstack((mask_g,mask_g,mask_g))
        
        yellow_lower_range = np.array([16,120,105])
        yellow_upper_range = np.array([24,255,255])
        mask_y = cv2.inRange(rgb_resized, yellow_lower_range, yellow_upper_range)
        mask_y = mask_y.astype('bool')
        y_region = rgb_resized * np.dstack((mask_y,mask_y,mask_y))

        redb_lower_range = np.array([0,100,100])
        redb_upper_range = np.array([15,255,255])
        mask_rb = cv2.inRange(rgb_resized, redb_lower_range, redb_upper_range)
        mask_rb = mask_rb.astype('bool')
        rb_region = rgb_resized * np.dstack((mask_rb,mask_rb,mask_rb))
            
        Green=0
        for y in range(0,width_resize):
            for x in range(0,height_resize):
                if mask_g[x][y] > 0:
                    Green=Green+1
        
        Yellow=0
        for y in range(0,width_resize):
            for x in range(0,height_resize):
                if mask_y[x][y] > 0:
                    Yellow=Yellow+1
        
        Reddish_Brown=0
        for y in range(0,width_resize):
            for x in range(0,height_resize):
                if mask_rb[x][y] > 0:
                    Reddish_Brown=Reddish_Brown+1
        
        G_Ratio = (Green/(Green+Yellow+Reddish_Brown))*100
        Y_Ratio = (Yellow/(Green+Yellow+Reddish_Brown))*100
        RB_Ratio = (Reddish_Brown/(Green+Yellow+Reddish_Brown))*100

        dataset = pd.read_csv(r"C:\Users\acer\Downloads\Pineapple_Data3.csv")
        feature = dataset.values[:,0:3]
        label = dataset.values[:,3:4]
        classifier = tree.DecisionTreeClassifier()
        classifier = classifier.fit(feature,label)
        group = classifier.predict([[G_Ratio,Y_Ratio,RB_Ratio]])
        
        dataset2 = pd.read_csv(r"C:\Users\acer\Downloads\Pineapple_Taste2.csv")
        feature2 = dataset2.values[:,0:4]
        label2 = dataset2.values[:,4:5]
        name_feature2 = ['Green', 'Yellow', 'ReddishBrown', 'Group']
        name_label2 = ['Not suitable for comsumption',
                    'Acidic','Sweet and sour','Sweet']
        classifier2 = tree.DecisionTreeClassifier()
        classifier2 = classifier2.fit(feature2,label2)
        taste = classifier2.predict([[G_Ratio,Y_Ratio,RB_Ratio,group]])
        num_taste = int(taste)
        
        print(image,name_label2[num_taste-1])
        return (image,name_label2[num_taste-1])
        
