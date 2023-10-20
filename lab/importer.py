import numpy as np
import cv2
import os
import random
from natsort import natsorted

class Dataset2D():
    def __init__(self, path):
        self.images_path_list=[]
        self.keys_path_list=[]
        data_folder=r"{}".format(path)
        images_folder=r"{}/images/".format(data_folder)
        self.keys_paths=r"{}/speed.txt".format(data_folder)
        for images_path in natsorted(os.listdir(images_folder)):
            images_path = os.path.join(images_folder, images_path)
            self.images_path_list.append(images_path)

    def Import(self,size=(28, 28)):
        self.X = []
        self.y = []
        for images_path in self.images_path_list:
            print("processing {}".format(images_path))
            img = cv2.imread(images_path)
            img = cv2.resize(img, size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.X.append(img)
        self.X = np.array(self.X)
        f=open(self.keys_paths,"r")
        lines = f.readlines()
        for line in lines:
            key = line.strip()
            print("processing {}".format(key))
            self.y.append(key)
        return self.X, self.y


def Split2D(X,y,test_size=0.2):
        len_y=len(y)
        y_test_number=(len_y*test_size)
        y_test_number=int(y_test_number)    
        y_test=y[: y_test_number]
        y_train=y[y_test_number :]

        x_shape=X.shape
        x_test_number=(x_shape[0]*test_size)
        x_test_number=int(x_test_number)    
        x_test=X[: x_test_number]
        x_train=X[x_test_number :]
        return x_train,x_test,y_train,y_test

def Normalize2D(dataset):
    img_row=dataset.shape[1]
    img_col=dataset.shape[2]
    channel=dataset.shape[3]
    dataset=dataset.astype('float32')/255
    dataset=dataset.reshape(dataset.shape[0],img_row, img_col, channel)
    return dataset 

def Concatenate2D(dataset1,dataset2):
    return np.concatenate((dataset1,dataset2),axis=0)

def Label_count2D(label,label_name,categories):
    for i in range(len(categories)):
        label=list(label)
        print(label_name,categories[i],"==>",(label.count(i)/len(label))*100)

def Save(folder_name,dataset,dataset_name):
    current_dir=os.getcwd()
    save_folder=r"{}/DataFolder/".format(current_dir)
    if os.path.exists(save_folder)==False:os.mkdir(save_folder) 
    save_dir=r"{}/{}".format(save_folder,folder_name)
    if os.path.exists(save_dir)==False:os.mkdir(save_dir) 
    save_path=r"{}/{}".format(save_dir,dataset_name)
    np.save(save_path, dataset)

def Load(folder_name,dataset):
    current_dir=os.getcwd()
    save_folder=r"{}/DataFolder/".format(current_dir)
    save_dir=r"{}/{}".format(save_folder,folder_name)
    dataset_name=dataset+".npy"
    file_path=r"{}/{}".format(save_dir,dataset_name)
    if os.path.exists(file_path):data=np.load(file_path) 
    return data


