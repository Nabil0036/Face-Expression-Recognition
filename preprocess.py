import os
import random
import numpy as np
import cv2
from sklearn.preprocessing import LabelBinarizer

class Preprocess:

    def __init__(self,train_path,val_path):
        self.train_path = train_path
        self.val_path = val_path
        self.train_X =[]
        self.val_X = []
        self.test_X = []
        self.train_Y =[]
        self.val_Y = []
        self.test_Y = []
        self.valtest_X = []
        self.valtest_Y = []
        

    def shuffle(self,a,b,seed=42):
        self.a = a
        self.b = b
        self.seed = seed
        c = list(zip(self.a,self.b))
        random.shuffle(c)
        self.a, self.b = zip(*c)

        return self.a, self.b 

    def one_hot(self, y):
        self.y = y
        lb = LabelBinarizer()

        self.y = lb.fit_transform(self.y)
        self.y = np.array(self.y)
        
        return self.y 


    def image_and_label(self):
        self.emotes_train = os.listdir(self.train_path)
        self.emotes_test = os.listdir(self.val_path)

        for e_train in self.emotes_train:
            images = os.listdir(os.path.join(self.train_path,e_train))
            for img in images:
                img_path = os.path.join(self.train_path,e_train,img)
                train_img = cv2.imread(img_path,0)
                self.train_X.append(train_img)
                self.train_Y.append(e_train)

        for e_test in self.emotes_test:
            images = os.listdir(os.path.join(self.val_path,e_test))
            for img in images:
                img_path = os.path.join(self.val_path,e_test,img)
                test_img = cv2.imread(img_path,0)
                self.valtest_X.append(test_img)
                self.valtest_Y.append(e_test)
        
        l = len(self.valtest_X)
        l = int(l/2)
        print(l)
        self.train_X, self.train_Y = self.shuffle(self.train_X, self.train_Y)
        self.valtest_X, self.valtest_Y = self.shuffle(self.valtest_X, self.valtest_Y)

        self.val_X = self.valtest_X[:l]
        self.val_Y = self.valtest_Y[:l]
        ##
        print(len(self.val_X))
        ##
        self.test_X = self.valtest_X[l:]
        self.test_Y = self.valtest_Y[l:]

        self.train_Y = self.one_hot(self.train_Y)
        self.val_Y = self.one_hot(self.val_Y)
        self.test_Y = self.one_hot(self.test_Y)

        self.train_X = np.array(self.train_X).reshape(-1,48,48)
        self.val_X = np.array(self.val_X).reshape(-1,48,48)
        self.test_X = np.array(self.test_X).reshape(-1,48,48)

        self.train_X = self.train_X/255
        self.val_X = self.val_X/255
        self.test_X = self.test_X/255

        self.train_X = self.train_X.reshape(28821,48,48,1)
        self.val_X = self.val_X.reshape(3533,48,48,1)
        self.test_X = self.test_X.reshape(3533,48,48,1)

        return self.train_X, self.train_Y, self.val_X, self.val_Y, self.test_X, self.test_Y