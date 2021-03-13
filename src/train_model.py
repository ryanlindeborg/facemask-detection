import tensorflow as tf
import numpy as np
import random as rn

import multiprocessing
import cv2
import os

from keras.layers import Dense, Activation, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Set random seed
os.environ['PYTHONHASHSEED'] = '0'
SEED = 1
np.random.seed(SEED)
rn.seed(SEED)
tf.random.set_seed(SEED)
num_CPU = 1
num_GPU = 0
max_cores = multiprocessing.cpu_count()

# Use the file path where your dataset is stored
data_path = './data/'
categories = os.listdir(data_path)
labels = [i for i in range(len(categories))]
label_dict = dict(zip(categories, labels))

# Make list for data and target
img_size = 150
data = []
target = []
for category in categories:
    folder_path = os.path.join(data_path, category)
    img_names = os.listdir(folder_path)

    for img_name in img_names:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (img_size, img_size))
            data.append(resized)
            target.append(label_dict[category])

        except Exception as e:
            print("Exception: ", e)

# Normalize data
data = np.array(data) / 255.0

# Reshaping of data
data = np.reshape(data, (data.shape[0], img_size, img_size, 1))
target = np.array(target)
new_target = np_utils.to_categorical(target)

# Build a neural network
model = Sequential()
model.add(Conv2D(200, (3, 3), input_shape=data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(100, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()

# Split data in train and test
train_data, test_data, train_target, test_target = train_test_split(data, new_target, test_size=0.15, random_state=42)

# Train model and save checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='min', restore_best_weights=True)
checkpoint = ModelCheckpoint('./models/checkpoints/model-{epoch:03d}.model', monitor='val_loss', verbose=1,
                             save_best_only=True, mode='auto')
history = model.fit(train_data, train_target, epochs=20, callbacks=[checkpoint, early_stopping], validation_split=0.2,
                    use_multiprocessing=True, workers=12)

# Get training and validation metrics
plt.plot(history.history['acc'], 'r', label='training accuracy')
plt.plot(history.history['val_acc'], 'b', label='validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

# Evaluate model on test data
print(model.evaluate(test_data, test_target))

# Save model
model.save('./models/final/mask_model.h5')
