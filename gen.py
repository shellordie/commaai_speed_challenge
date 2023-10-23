import cv2
import os
from natsort import natsorted
from PIL import Image
import numpy as np
import tensorflow as tf

def _gen_dataset():
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
                frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img=Image.fromarray(frame)
                img.save(path_to_check)
                nbr+=1
            else:
                print("exist",path_to_check)
                nbr+=1
    cap.release()
    cv2.destroyAllWindows()

def _load_model(model_name):
    current_dir=os.getcwd()
    model_dir=r"{}/lab/SavedModel".format(current_dir)
    model_path="{}/{}".format(model_dir,model_name)
    if os.path.exists(model_path):
        model=tf.keras.models.load_model(model_path)
        return model

def _gen_pred():
    path=r"C:\Users\charleslf\Downloads\Video\test.mp4"
    save_path=r"{}/pred.txt".format(os.getcwd())
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

#_gen_dataset()
_gen_pred()
