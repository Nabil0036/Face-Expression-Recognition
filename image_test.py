import cv2
import numpy as np
from tensorflow.keras.models import load_model
from model import Preprocess
import os

model = load_model("model.h5")
image = cv2.imread("qwe.jpg")
prep = Preprocess()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

boxes = face_cascade.detectMultiScale(image, 1.3, 5)
box = boxes[0]
(x,y,w,h) = box
image = cv2.rectangle(image, (x,y),(x+w,y+h),(255,0,0),2)
roi = image[y:y+h,x:x+w]
final = prep.preprocess(roi)
p = model.predict(final)


print(final.shape)
print(np.argmax(p))
print(os.listdir("dataset/train/")[np.argmax(p)])
cv2.imshow("image",roi)
cv2.waitKey(0)
cv2.destroyAllWindows()