from importer import Load
import tensorflow as tf
from model import Pilot_net 
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from config import model_name,groundtruth2D_data 
import os

def _save_model(model,name):
    current_dir=os.getcwd()
    save_dir=r"{}/SavedModel".format(current_dir)
    model_save_dir="{}/{}".format(save_dir,name)
    if os.path.exists(save_dir)==False:
        os.mkdir(save_dir)
    model.save(model_save_dir)
    print("model saved")

def _load_model(model,name):
    current_dir=os.getcwd()
    save_dir=r"{}/SavedModel".format(current_dir)
    model_save_dir="{}/{}".format(save_dir,name)
    if os.path.exists(model_save_dir):
        model=tf.keras.models.load_model(model_save_dir) 
        print("model loaded")
    return model

def train(numdata,model_name,epochs):
    x_train=Load(numdata,"x_train")
    y_train=Load(numdata,"y_train")
    x_test=Load(numdata,"x_test")
    y_test=Load(numdata,"y_test")
    input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3])
    print("input shape",input_shape)
    model=Pilot_net(input_shape)
    model.compile(optimizer='rmsprop',
                  loss="mse",
                  loss_weights=2.,
                  metrics="mae")
    model=_load_model(model,model_name)
    model.summary()
    model.fit(x_train,y_train,epochs=epochs,batch_size=32)
    test_loss,test_acc=model.evaluate(x_test,y_test,verbose=2)
    f=open("acc.txt","r")
    last_acc=f.readline()
    print("\n Last Test accuracy:",last_acc)
    print("\n Current Test accuracy:",test_acc)
    user_input=input("do you want to save this model ? y/n :")
    if user_input=="y":
        f=open("acc.txt","w")
        f.write(str(test_acc))
        f.close()
        _save_model(model,model_name)
    else:
        exit(1)

print("Num GPU availaible :",len(tf.config.list_physical_devices('GPU')))
numdata=groundtruth2D_data
model_name=model_name
train(numdata,model_name,epochs=5)

