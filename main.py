import cv2
import os
from PIL import Image
import numpy as np
import time
def _play_video():
    time.sleep(10)
    video_path=r"C:\Users\charleslf\Downloads\Video\test.mp4"
    save_path=r"{}/pred.txt".format(os.getcwd())
    nbr=0
    f=open(save_path,"r")
    lines=f.readlines()
    cap=cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret,frame=cap.read()
        if ret ==True:
            display="speed ={} m/s".format(round(float(lines[nbr].strip()),1))
            font=cv2.FONT_HERSHEY_SIMPLEX
            frame=cv2.putText(frame,display,(50,50),font,1,(0,0,255),2)
            cv2.imshow('Frame',frame)
            nbr+=1
            if cv2.waitKey(25)& 0xFF==ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

_play_video()
