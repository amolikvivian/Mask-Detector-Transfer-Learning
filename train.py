import os
import numpy as np 
from imutils import paths
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input

from keras.layers import Dense, Flatten, Input
from keras.layers import AveragePooling2D, Dropout


DATASET_PATH = '/dataset/'

#Storing path of each image in one list
imagePathList = list(paths.list_images('DATASET_PATH'))

#Initializing data and labels list
data = []
labels = []

for imagePath in imagePathList:
	
	#Extracting class label from each image
	label = imagePath.split(os.path.sep)[-2]

	#Loading image and setting size to (224x224)
	image = load_img(imagePath, target_size(224, 224))
	
	#Converting image to array and applying MobileNetV2 preprocess
	image = img_to_array(image)
	image = preprocess_input(image)

	#Appending to data and labels list
	data.append(image)
	labels.append(label)

#Converting data and labels to numpy arrays
data = np.array(data)
labels = np.array(labels)

#One-Hot-Encoding Labels
label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(labels)
labels = to_categorical(labels)

#Splitting dataset into Train:Test - 9:1
(trainData, testData, trainLabels, testLabels) = train_test_split(data, labels,
	test_size=0.1, stratify=labels, random_state=42)

trainDataGenerator = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")

#Building our Transfer Learning Model

#Loading MobileNetV2 Model as a Base Model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

#Extracting output layers from Base Model
x = base_model.output

#Adding layers to MobileNet model
model = AveragePooling2D(pool_size=(7, 7))(x)

model = Flatten()(model)

model = Dense(128, activation="relu")(model)
model = Dense(64, activation="relu")(model)

model = Dropout(0.5)(model)

model = Dense(2, activation="softmax")(model)

#Creating model with inputs as base model and outputs as the model we prepared
finalModel = Model(inputs=base_model.input, outputs=model)

#Making all layers in the base model as untrainable
for layer in base_model.layers:
	layer.trainable = False

#Compiling model
model.compile(loss='binary_crossentropy', optimizer='Adam', 
	metrics=['accuracy'])

#Training Model
history = model.fit(aug.flow(trainData, trainLabels, batch_size=26),
	steps_per_epoch=len(trainData) // 26,
	validation_split=0.2, epochs=15)

MODEL_SAVE_PATH = 'savedModels/'
PLOT_SAVE_PATH = 'plot/'

print('[STATUS] Saving model')
model.save(MODEL_SAVE_PATH, save_format='h5')

#Plotting model accuracy and loss
N = 15
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy"))
plt.savefig('PLOT_SAVE_PATH')