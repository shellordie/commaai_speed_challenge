from importer import Load,Label_count2D
import matplotlib.pyplot as plt
from config import classes,groundtruth2D_data

x_train=Load(groundtruth2D_data,"x_train")
y_train=Load(groundtruth2D_data, "y_train")
x_test=Load(groundtruth2D_data,"x_test")
y_test=Load(groundtruth2D_data,"y_test")

def _dataset_info(X,y,X_name,y_name):
    print("{} shape ==> {} ".format(X_name,X.shape))
    print("{} shape ==> {}".format(y_name,len(y)))
    print("-------------------------------------------")

print("-------------------------------------")
_dataset_info(x_train,y_train,"x_train","y_train")
_dataset_info(x_test,y_test,"x_test","y_test")
print("-----------------------------------")
print(X.shape)
print(y.shape)

