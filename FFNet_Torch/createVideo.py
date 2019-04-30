import cv2
import  numpy as np
import glob
import scipy.io as sio

img_arr = []
temp1 = sio.loadmat('/data/Guha/Tour20/Result/ET8.mat')
summary = temp1['summary'][0]
for i in range(len(summary)):
    if(summary[i] ==1):
        #file = glob.('/data/Guha/Tour20/frames/BF6.mp4/frame{}.jpg'.format(i))
        img = cv2.imread('/data/Guha/Tour20/frames/ET8.mp4/frame{}.jpg'.format(i))

        height, width, layers = img.shape
        size = (width, height)
        img_arr.append(img)

#out = cv2.VideoWriter('AW7.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, size))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('ET8.mp4',fourcc, 20.0, size)

for i in range(len(img_arr)):
    out.write(img_arr[i])
out.release()