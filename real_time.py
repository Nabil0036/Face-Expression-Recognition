import cv2
import os
import time
from tensorflow.keras.models import load_model
from model import Preprocess


model = load_model("model.h5")
cap = cv2.VideoCapture(1)
time.sleep(2)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
prep = Preprocess()

while True:
    ret, frame = cap.read()
    boxes = face_cascade.detectMultiScale(frame, 1.3,5)
    check_tuple = type(boxes) is tuple
    #print(boxes)
    if len(boxes)>=1 and not check_tuple:
        box = boxes[0]
        x,y,w,h = box[0],box[1],box[2],box[3]
        tup_box = (x,y,w,h)
        #print(tup_box)
        if w>120 and h >120:
            face = frame[y:y+h,x:x+w]
            final = prep.preprocess(face)
            pred = model.predict(final)
            l = pred.argmax()
            label = os.listdir("dataset/train")[l]
            print(label)
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            frame = cv2.putText(frame, label, (x-20,y-20), cv2.FONT_HERSHEY_SIMPLEX, 1,  
                 (0,255,0), 2, cv2.LINE_AA) 
        else:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.imshow('Face Expression Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        cv2.imshow('Face Expression Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    
    

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows() 