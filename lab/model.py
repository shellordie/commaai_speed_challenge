import tensorflow as tf 
import keras
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dropout,Input
from tensorflow.keras.layers import Softmax,Flatten,Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import ResNet50 

def _maxpool2d(y):
    y=MaxPooling2D(pool_size=1,strides=1)(y)
    return y

def conv2d(y,filters):
    y=Conv2D(filters,kernel_size=2,activation='relu')(y)
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

base_model=ResNet50(include_top=False)
base_model.trainable=False

def Pilot_net(input_shape):
    inputs=Input(shape=input_shape)
    y=data_aug(inputs)
    y=data_aug2(y)
    y=base_model(y,training=False)
    y=conv2d(y,64)
    y=_maxpool2d(y)
    y=conv2d(y,128)
    y=_maxpool2d(y)
    y=conv2d(y,256)
    y=_maxpool2d(y)
    y=Dropout(.2)(y)
    y=Flatten()(y)
    y=Dense(16, activation="relu")(y)
    y=Dense(32, activation="relu")(y)
    y=Dense(64, activation="relu")(y)
    outputs=Dense(1,activation='relu')(y)
    model=keras.Model(inputs,outputs,name="Pilot_net")
    return model

model=Pilot_net(input_shape=(200,200,3))
model.summary()


