import cv2
import os
from natsort import natsorted
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 

path=r"C:\Users\charleslf\Downloads\Video\train.mp4"
save_path=r"{}/data/images/".format(os.getcwd())
cap=cv2.VideoCapture(path)
nbr=0
while cap.isOpened():
    ret,frame=cap.read()
    if ret ==True:
        path_to_check=r"{}/img{}.png".format(save_path,nbr)
        if os.path.exists(path_to_check)==False:
            print("processing ",path_to_check)
            plt.imsave(path_to_check,frame)
            nbr+=1
        else:
            print("exist",path_to_check)
            nbr+=1
cap.release()
cv2.destroyAllWindows()
