import cv2
import os
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from keras.models import load_model

# Use the file path where your dataset is stored
data_path = './data/'
categories = os.listdir(data_path)
labels = [i for i in range(len(categories))]
label_dict = dict(zip(categories, labels))
print(label_dict)
print(categories)
print(labels)

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

# Saving the files
np.save('./objects/data', data)
np.save('./objects/target', new_target)

# Build a neural network
data = np.load('./objects/data.npy')
target = np.load('./objects/target.npy')
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
train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.1)

# Train model and save checkpoint
checkpoint = ModelCheckpoint('./model_checkpoints/model-{epoch:03d}.model', monitor='val_loss', verbose=0,
                             save_best_only=True, mode='auto')
history = model.fit(train_data, train_target, epochs=20, callbacks=[checkpoint], validation_split=0.2)

# Load the best model
model = load_model('./model/model_checkpoints/model-005.model')
faceCascade = cv2.CascadeClassifier('./opencv/data/HaarCascade/haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)
labels_dict = {0: 'NO MASK', 1: 'MASK'}
color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        face_img = gray[y:y + w, x:x + h]
        resized = cv2.resize(face_img, (img_size, img_size))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, img_size, img_size, 1))
        result = model.predict(reshaped)

        label = np.argmax(result, axis=1)[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color_dict[label], 2)
        cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('Video', frame)
    key = cv2.waitKey(1)

    if key == 27:
        break

cv2.destroyAllWindows()
video_capture.release()
