import tensorflow as tf 
import keras
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dropout,Input
from tensorflow.keras.layers import Softmax,Flatten,Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2 

def _maxpool2d(y):
    y=MaxPooling2D(pool_size=1,strides=1)(y)
    return y

def _squeeze(y,filters):
    y=Conv2D(filters,kernel_size=1,activation='relu')(y)
    return y

def _e1x1(y,filters):
    y=Conv2D(filters,kernel_size=1,activation="relu")(y)
    return y

def _e3x3(y,filters):
    y=Conv2D(filters,kernel_size=3,activation='relu')(y)
    return y

def _expand(y,filters):
    #y=e1x1(y,filters)
    #y=e1x1(y,filters)
    #y=e1x1(y,filters)
    #y=e3x3(y,filters)
    #y=e3x3(y,filters)
    #y=e3x3(y,filters)
    y=_e1x1(y,filters)
    y=_e3x3(y,filters)
    return y

def _fire(y,filters,name):
    y=_squeeze(y,filters)
    y=_expand(y,filters)
    return y

data_aug=tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomFlip("vertical"),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomRotation(0.4),
    layers.RandomRotation(0.6),
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomFlip("vertical"),
    layers.RandomFlip("horizontal"),
    layers.Resizing(224,224),
    layers.Rescaling(1./255),
    ])

def data_aug2(y):
    y=tf.image.flip_left_right(y)
    y=tf.image.adjust_brightness(y,delta=0.1)
    y=tf.image.central_crop(y,central_fraction=0.5)
    y=tf.image.adjust_brightness(y,delta=0.2)
    return y

base_model=MobileNetV2(include_top=False)
base_model.trainable=False

def Pilot_net(input_shape):
    inputs=Input(shape=input_shape)
    y=data_aug(inputs)
    y=data_aug2(y)
    y=base_model(y,training=False)
    #y=Conv2D(filters=96,kernel_size=7,strides=2)(y)
    #y=_fire(y,filters=128,name="fire2")
    #y=_fire(y,filters=128,name="fire3")
    #y=_fire(y,filters=256,name="fire4")
    #y=_maxpool2d(y)
    #y=_fire(y,filters=256,name="fire5")
    y=_fire(y,filters=384,name="fire6")
    #y=_fire(y,filters=384,name="fire7")
    y=_maxpool2d(y)
    #y=_fire(y,filters=512,name="fire8")
    #y=_fire(y,filters=512,name="fire9")
    y=Dropout(.5)(y)
    y=Conv2D(filters=10,kernel_size=1,strides=1)(y)
    y=Flatten()(y)
    y=Dense(64, activation="relu")(y)
    outputs=Dense(1,activation='relu')(y)
    model=keras.Model(inputs,outputs,name="Pilot_net")
    return model

model=Pilot_net(input_shape=(224,224,3))
model.summary()


