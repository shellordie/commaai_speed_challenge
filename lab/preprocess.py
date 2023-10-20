from importer import Dataset2D,Save,Load,Normalize2D,Label_count2D
from importer import Concatenate2D,Split2D
from importer import Concatenate2D,Split2D
import matplotlib.pyplot as plt
from config import model_name,groundtruth2D_data
import os

def _dataset_info(X,y,X_name,y_name):
    print("{} shape ==> {} ".format(X_name,X.shape))
    print("{} shape ==> {}".format(y_name,len(y)))
    print("-------------------------------------------")

def _save(numdata,x_train,x_test,y_train,y_test):
    Save(numdata,x_train,"x_train")
    Save(numdata,x_test,"x_test")
    Save(numdata,y_train,"y_train")
    Save(numdata,y_test,"y_test")


def _concatenate(dataset1,dataset2):return Concatenate2D(dataset1,dataset2)

def _is_groundtruth():
    current_path=os.getcwd()
    path_to_check=r"{}/DataFolder/groundtruth2D".format(current_path)
    if os.path.exists(path_to_check)==True:
        return True
    else:
        return False

def import_dataset(datapath):
    dataset=Dataset2D(datapath)
    X,y=dataset.Import(size=(199,199))
    return X,y

def data_loader():
    new_path=r"C:\Users\charleslf\Desktop\commaai_speed_challenge\data"
    print("datapath = {}\n".format(new_path))
    user_input=input("proceed ? (y/n):") 
    if user_input == "y":
        X,y=import_dataset(new_path)
    else:
        exit(1)
    if _is_groundtruth()==True:
        print("merging with groundthruth...")
        ground_x=Load("groundtruth2D","ground_X")
        ground_y=Load("groundtruth2D","ground_y")
        X=_concatenate(ground_x,X)
        y=_concatenate(ground_y,y)
    Save("groundtruth2D",X,"ground_X")
    Save("groundtruth2D",y,"ground_y")
    print("saved in the groundtruth")
    return X,y

def data_split_save(X,y,data_name):
    """ the preprocessor """
    x_train,x_test,y_train,y_test=Split2D(X,y)
    x_train=Normalize2D(x_train)
    x_test=Normalize2D(x_test)
    _save(data_name,x_train,x_test,y_train,y_test)
    _dataset_info(X,y,"x","y")
    _dataset_info(x_train,y_train,"x_train","y_train")
    _dataset_info(x_test,y_test,"x_test","y_test")


data_name=groundtruth2D_data
X,y=data_loader()
data_split_save(X,y,data_name)


