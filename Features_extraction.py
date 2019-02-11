#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
# display plots in this notebook
#%matplotlib inline
# filter out the warnings
import warnings
warnings.filterwarnings('ignore')
from numpy import load
import numpy as np
import cv2 as cv
from PIL import Image
data = np.load('eye_tracker_train_and_val.npz')
cascades_path = '/usr/share/opencv/haarcascades/'
#face_cascade = cv2.CascadeClassifier(cascades_path + 'haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier(cascades_path + 'haarcascade_eye.xml')
new_path = '/Users/lindamokrane/Desktop/Insight/Script/xml/'
face_cascade = cv2.CascadeClassifier(new_path + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(new_path + 'haarcascade_eye.xml')
import time

def current_time():
    return int(round(time.time() * 1000))

def extract_face_features(face, img, gray):
    [x,y,w,h] = face
    roi_gray = gray[y:y+h, x:x+w]
    face_image = np.copy(img[y:y+h, x:x+w])
    
    eyes = eye_cascade.detectMultiScale(roi_gray)
    eye_images = []
    for (ex,ey,ew,eh) in eyes:
        eye_images.append(np.copy(img[y+ey:y+ey+eh,x+ex:x+ex+ew]))
                
    roi_color = img[y:y+h, x:x+w]
    for (ex,ey,ew,eh) in eyes:
        return face_image, eye_images

def get_face_grid(face, frameW, frameH, gridSize):
    faceX,faceY,faceW,faceH = face
  #  print("x: {} y: {} w: {} h: {} ", face[0], face[1], face[2], face[3])
   # aceX,faceY,faceW,faceH = face
    return faceGridFromFaceRect(frameW, frameH, gridSize, gridSize, faceX, faceY, faceW, faceH, False)
   # return faceGridFromFaceRect(frameW, frameH, gridSize, gridSize, faceX, faceY, faceW, faceH, False)

def extract_image_features(full_img_path):
    img = cv2.imread(full_img_path)
    start_ms = current_time()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_detections = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    faces = []
    face_features = []
    for [x,y,w,h] in face_detections:
        face = [x, y, w, h]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        face_image, eye_images = extract_face_features(face, img, gray)
        face_grid = get_face_grid(face, img.shape[1], img.shape[0], 25)
        
        faces.append(face)
        face_features.append([face_image, eye_images, face_grid])
    
    duration_ms = current_time() - start_ms
    print("Face and eye extraction took: ", str(duration_ms / 1000) + "s")
    
    return img, faces, face_features

gridSize = 25
####################
# Given face detection data, generate face grid data.
#
# Input Parameters:
# - frameW/H: The frame in which the detections exist
# - gridW/H: The size of the grid (typically same aspect ratio as the
#     frame, but much smaller)
# - labelFaceX/Y/W/H: The face detection (x and y are 0-based image
#     coordinates)
# - parameterized: Whether to actually output the grid or just the
#     [x y w h] of the 1s square within the gridW x gridH grid.
########
def faceGridFromFaceRect(frameW, frameH, gridW, gridH, labelFaceX, labelFaceY, labelFaceW, labelFaceH, parameterized):

    scaleX = gridW / frameW
    scaleY = gridH / frameH
    
    if parameterized:
      labelFaceGrid = np.zeros(4)
    else:
      labelFaceGrid = np.zeros(gridW * gridH)
    
    grid = np.zeros((gridH, gridW))

    # Use one-based image coordinates.
    xLo = round(labelFaceX * scaleX)
    yLo = round(labelFaceY * scaleY)
    w = round(labelFaceW * scaleX)
    h = round(labelFaceH * scaleY)

    if parameterized:
        labelFaceGrid = [xLo, yLo, w, h]
    else:
        xHi = xLo + w
        yHi = yLo + h

        # Clamp the values in the range.
        xLo = int(min(gridW, max(0, xLo)))
        xHi = int(min(gridW, max(0, xHi)))
        yLo = int(min(gridH, max(0, yLo)))
        yHi = int(min(gridH, max(0, yHi)))

        faceLocation = np.ones((yHi - yLo, xHi - xLo))
        grid[yLo:yHi, xLo:xHi] = faceLocation

        # Flatten the grid.
        grid = np.transpose(grid)
        labelFaceGrid = grid.flatten()
        
    return labelFaceGrid
######################
def set_title_and_hide_axis(title):
    plt.title(title)
    plt.axes().get_xaxis().set_visible(False)
    plt.axes().get_yaxis().set_visible(False)

def render_face_grid(face_grid):
    face_grid = np.asarray(face_grid)
    to_print = np.copy(face_grid)
    print("size", to_print.shape)
    print("size" , render_face_grid)
    print("size" , face_grid)
    result_image = np.copy(to_print).reshape(25, 25).transpose()
    plt.figure()
    set_title_and_hide_axis('Face grid')
#     print(result_image.shape)
    plt.imshow(result_image)

def show_extraction_results(img, faces, face_features):
    face_features = np.asarray(face_features)
    plt.figure(figsize=(10,10))
    set_title_and_hide_axis('Original image and extracted features')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), interpolation="bicubic")

    for i, face in enumerate(faces):
        print('Face #' + str(i))
        #print('i', face, i)
        face_image, eye_images, face_grid = face_features[i]
       # face_image = face_features[i]
      #  eye_images = face_features[i]
     #   face_grid = face_features[i]                          
        print("face_features" , face_features[i])
        print("face_grid" , face_grid)
        plt.figure()
        set_title_and_hide_axis('Extracted face image')
        plt.imshow(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB), interpolation="bicubic")
        plt.figure()
        print("face_grid" ,face_grid)
        #print('face image after extraction')
        render_face_grid(face_grid)

        for eye_image in eye_images:
            plt.figure()

            #print('eye image after extraction')
            set_title_and_hide_axis('Extracted eye image')
            plt.imshow(cv2.cvtColor(eye_image, cv2.COLOR_BGR2RGB), interpolation="bicubic")
#######################
img, faces, face_features = extract_image_features('img_0.jpg')
show_extraction_results(img, faces, face_features)
#########################




