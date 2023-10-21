import cv2
import os
from natsort import natsorted
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf

path=r"C:\Users\charleslf\Downloads\Video\test.mp4"
save_path=r"{}/pred.txt".format(os.getcwd())
cap=cv2.VideoCapture(path)
nbr=0

def _load_model(model_name):
    current_dir=os.getcwd()
    model_dir=r"{}/lab/SavedModel".format(current_dir)
    model_path="{}/{}".format(model_dir,model_name)
    if os.path.exists(model_path):
        model=tf.keras.models.load_model(model_path)
        return model

model=_load_model("speed_model_003")
print("model loaded")

while cap.isOpened():
    ret,frame=cap.read()
    if ret ==True:
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (200,200))
        predictions=model.predict(np.array([frame]))
        frame=cv2.putText(frame,str(predictions[0][0]),(50,50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
        cv2.imshow('Frame',frame)
        f=open(save_path,"a")
        f.write(str(predictions[0][0]))
        f.write("\n")
        f.close()

        if cv2.waitKey(25)& 0xFF==ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
