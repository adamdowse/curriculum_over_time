import cv2
import numpy as np
import os
import argparse

ap = argparse.ArgumentParser()
#ap.add_argument('-d','--data',required=True,help='path to collection of images')
#ap.add_argument('-n','--name',required=True,help='name of the output video')
ap.add_argument('-f','--fps',required=False,help='FPS of the video output')

img_array = []

img_dir = '/project/Data/normal/'

for filename in os.listdir(img_dir):
    if filename.endswith('.jpg'):
        img = cv2.imread(filename)
        height,width,layers = img.shape
        size = (width,height)
        img_array.append(img)

if ap.fps is not None:
    fps = ap.fps
else:
    fps = 3
    
out = cv2.VideoWriter('normal.mp4',cv2.VideoWriter_fourcc(*'DIVX'),fps,size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
