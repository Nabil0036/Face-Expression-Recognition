from model import VGGNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np


data = np.load("train_val_test.npz")
train_X, train_Y, val_X, val_Y, test_X, test_Y = data["arr_0"], data["arr_1"], data["arr_2"], data["arr_3"], data["arr_4"], data["arr_5"]

vgg = VGGNet()
model = vgg.build()
opt = Adam(lr=1e-3)
early_stop = EarlyStopping(monitor='val_loss',patience=5)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
history = model.fit(train_X,train_Y,epochs=30,validation_data=(val_X,val_Y),callbacks=[early_stop])

model.save("model.h5")