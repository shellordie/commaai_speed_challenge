import tensorflow as tf
import os
from importer import Load
import numpy as np
import matplotlib.pyplot as plt
from config import model_name,groundtruth2D_data

def _load_model(model_name):
    current_dir=os.getcwd()
    model_dir=r"{}/SavedModel".format(current_dir)
    model_path="{}/{}".format(model_dir,model_name)
    if os.path.exists(model_path):
        model=tf.keras.models.load_model(model_path)
        return model

def test(model_name,numdata):
    model=_load_model(model_name)
    model.summary()
    x_test=Load(numdata,"x_test")
    y_test=Load(numdata,"y_test")
    for i in range(0,30): 
        print("----------------------------------------------")
        print("real speed",y_test[i])
        predictions=model.predict(np.array([x_test[i]]))
        print("predicted speed",predictions)
        plt.imshow(x_test[i])
        plt.show()

model_name=model_name
numdata=groundtruth2D_data
test(model_name,numdata)



