from preprocess import Preprocess
import os 
import numpy as np


file_name = "train_val_test.npz"

train_path = os.path.join("dataset","train")
test_path = os.path.join("dataset","validation")

p = Preprocess(train_path,test_path)

train_X, train_Y, val_X, val_Y, test_X, test_Y = p.image_and_label()

np.savez(file_name,train_X, train_Y, val_X, val_Y, test_X, test_Y)

print(len(train_X),len(val_X),len(test_X))