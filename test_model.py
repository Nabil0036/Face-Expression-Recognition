from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import numpy as np
import cv2


data = np.load("train_val_test.npz")
model = load_model("model.h5")
plot_model(model, to_file="model.png")

test_X, test_Y = data["arr_4"], data["arr_5"] 
predicted = model.predict_classes(test_X)
print(predicted[0])
g = test_X[0].reshape((48,48))
print(g)

score, acc = model.evaluate(test_X,test_Y)
print(score, acc)
cv2.imshow("image", g)
cv2.waitKey(0)
cv2.destroyAllWindows()
