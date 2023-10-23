import cv2
import os
from natsort import natsorted
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
import time

path=r"C:\Users\charleslf\Downloads\Video\test.mp4"
save_path=r"{}/pred.txt".format(os.getcwd())

def _load_model(model_name):
    current_dir=os.getcwd()
    model_dir=r"{}/lab/SavedModel".format(current_dir)
    model_path="{}/{}".format(model_dir,model_name)
    if os.path.exists(model_path):
        model=tf.keras.models.load_model(model_path)
        return model

def _gen_pred():
    model=_load_model("speed_model_004")
    print("model loaded")
    cap=cv2.VideoCapture(path)
    while cap.isOpened():
        ret,frame=cap.read()
        if ret ==True:
            frame_2_pred = cv2.resize(frame, (200,200))
            predictions=model.predict(np.array([frame_2_pred]))
            f=open(save_path,"a")
            f.write(str(predictions[0][0]))
            f.write("\n")
            f.close()
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

def _play_video():
    nbr=0
    f=open(save_path,"r")
    lines=f.readlines()
    cap=cv2.VideoCapture(path)
    while cap.isOpened():
        ret,frame=cap.read()
        if ret ==True:
            display="speed ={} m/s".format(round(float(lines[nbr].strip()),1))
            font=cv2.FONT_HERSHEY_SIMPLEX
            frame=cv2.putText(frame,display,(50,50),font,1,(0,0,255),2)
            cv2.imshow('Frame',frame)
            nbr+=1
            #time.sleep(0.03)
            if cv2.waitKey(25)& 0xFF==ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

#_gen_pred()
_play_video()
