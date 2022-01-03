#################################################  Contactless Mask Recongnition    #################################################

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.layers import AveragePooling2D,GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout,BatchNormalization
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2



# directory of the datasets

data_directory = "/cai_lab/thomas/Mask Detection/dataset" # the datasets path that contains the dataset folder with label image 2 classes
categories = ["with_mask", "without_mask"]  # label names
              
# preprocessing stage........................................................................................
# resizing, normalzing, one hot encoding
data = []
labels = []
for category in categories:
    path = os.path.join(data_directory, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)   # or normalizing (image/225.0)
        data.append(image)
        labels.append(category)

# perform one-hot encoding on the labels .......................................................................................
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
data = np.array(data, dtype="float16")
labels = np.array(labels)
# print ("datasets shape: {}".format(data.shape) )
​
# splitting training and testing set .......................................................................................
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)
predefining the necessary parameters
batch_size = 32  # 64
img_size = 224  
n_class = 2   # three major class 
epochs = 10
# Importing the pretrained MobileNetv2 model .......................................................................................
# load the MobileNetV2 network, without the fully connected layer
inputs = Input (shape = (img_size, img_size, 3))
mobileNet_model=MobileNet(weights='imagenet',include_top=False,  input_shape= (img_size,img_size,3)) 
# mobileNet_model.summary()
output_mobileNet = mobileNet_model.output
# output_mobileNet = mobilenet_model(inputs)
​
dense_layer=GlobalAveragePooling2D()(output_mobileNet)
dense_layer = Dense(128, activation = 'relu')(dense_layer)
dense_layer = Dropout(0.5)(dense_layer)
dense_layer = Dense(128, activation = 'relu')(dense_layer)
dense_layer = BatchNormalization()(dense_layer)
dense_layer = Dropout(0.5)(dense_layer)
dense_layer = Dense(64, activation = 'relu')(dense_layer)
dense_layer = Dense(n_class, activation = 'softmax')(dense_layer)
model = Model(inputs=mobileNet_model.input, outputs=dense_layer)
# final_model = Model ([inputs], final_output)
model.summary()



 # data augmentation for increasing the accuracy and prevent overfitting.......................................................................................
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")    

##### compiling and training the model .......................................................................................

optimizer = Adam(lr=1e-4)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
model = model.fit( aug.flow(trainX, trainY, batch_size=batch_size),
              steps_per_epoch=len(trainX) // batch_size,
              validation_data=(testX, testY),
              validation_steps=len(testX) // batch_size,
              epochs=epochs)
model.save("mask_detector.h5") # saving the model to disk

##### predicting and plotting the training accuracy and loss.......................................................................................

# make predictions on the testing set
predicting = model.predict(testX, batch_size=batch_size)
predicting = np.argmax(predicting, axis=1)
# to see the classification report
print(classification_report(testY.argmax(axis=1), predicting,target_names=lb.classes_))

# plotting .......................................................................................
N = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
