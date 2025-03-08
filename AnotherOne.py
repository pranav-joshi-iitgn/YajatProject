# import required modules
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import cv2

model = load_model("GnA.h5")
model.load_weights("GnA.weights.h5")
I=cv2.imread("103.jpg")
I=cv2.resize(I,(224,224))
X=np.array([I])
#X =np.resize(I,(None,244,244,3))
print(float(model(X)[0,0]))