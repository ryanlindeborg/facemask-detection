import numpy as np
import cv2
import os

from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Sequential


def preprocess_image(img, img_size):
    """Convert to gray scale, resize and normalize"""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (img_size, img_size))
    normalized = resized / 255

    return normalized


def preprocess_training_data(data, img_size):
    """Convert data (list of images) to numpy array and reshape data to ingest it to the model"""

    # Preprocess all images
    for i, img in enumerate(data):
        data[i] = preprocess_image(img, img_size)

    # Convert to numpy array
    data = np.array(data)

    # Reshaping data
    data = np.reshape(data, (data.shape[0], img_size, img_size, 1))

    return data


def get_labels_from_data_path(data_path):
    """Get categories from data folder, each subfolder in it represent a label"""

    categories = os.listdir(data_path)
    labels = [i for i in range(len(categories))]
    label_dict = dict(zip(categories, labels))

    return label_dict


def read_images_from_data_folder(data_path):
    """Read all images in subfolders of data image"""

    # Get labels
    label_dict = get_labels_from_data_path(data_path)

    data = []
    target = []
    for category in label_dict.keys():
        folder_path = os.path.join(data_path, category)
        img_names = os.listdir(folder_path)

        for img_name in img_names:
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            data.append(img)
            target.append(label_dict[category])

    return data, target


def read_images_and_image_paths_from_data_folder(data_path):
    """Read all images in subfolders of data image and return corresponding list of image paths in addition to image
    data"""

    img_paths = []
    # Get labels
    label_dict = get_labels_from_data_path(data_path)

    data = []
    target = []
    for category in label_dict.keys():
        folder_path = os.path.join(data_path, category)
        img_names = os.listdir(folder_path)

        for img_name in img_names:
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            data.append(img)
            target.append(label_dict[category])
            img_paths.append(img_path)

    return data, target, np.array(img_paths)


def prepare_data_to_model(data, target, img_size=150):
    """Prepare all images in data folder to feed it in the model"""

    data = preprocess_training_data(data, img_size)
    target = np.array(target)
    target = to_categorical(target)

    return data, target


def create_model(input_shape):
    """Create arquitecture of neural network"""

    model = Sequential()
    model.add(Conv2D(200, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(100, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', AUC(name='auc')])
    model.summary()

    return model


def fit_model(data, target, model, n_epochs=20, model_checkpoint=True, **kwargs):
    """Split data in train, validation and test and train model"""

    # Set early stopping as a callback function
    early_stopping = EarlyStopping(monitor='val_auc',
                                   patience=5,
                                   verbose=1,
                                   mode='max',
                                   restore_best_weights=True)

    # Create model checkpoints as a callback function
    checkpoint = ModelCheckpoint('./models/checkpoints/model-{epoch:03d}.model',
                                 monitor='val_auc',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max')

    if model_checkpoint:
        callbacks_funcs = [checkpoint, early_stopping]
    else:
        callbacks_funcs = early_stopping

    # Fit model
    history = model.fit(data,
                        target,
                        epochs=n_epochs,
                        callbacks=callbacks_funcs,
                        validation_split=0.2,
                        **kwargs)

    return model, history


def draw_bounding_boxes_and_confidences(frame, locs, confidences, change_color=True):
    """Draw bounding boxes and confidence in image"""

    # Keep original image intact
    copy_frame = np.copy(frame)

    for box, confidence in zip(locs, confidences):
        (start_x, start_y, end_x, end_y) = box

        # Format confidence from probability to percentage
        text = f"{confidence * 100:.2f}%"

        # Shift down 10 px y coordinate in case face detection occurs at the very top of the image
        y = start_y - 10 if start_y - 10 > 10 else start_y + 10

        # Change color
        if change_color:
            if confidence > 0.5:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
        else:
            color = (46, 134, 193)

        # Draw rectangle in image
        copy_frame = cv2.rectangle(img=copy_frame,
                                   pt1=(start_x, start_y),
                                   pt2=(end_x, end_y),
                                   color=color,
                                   thickness=2)

        # Put text with confidence in image
        copy_frame = cv2.putText(img=copy_frame,
                                 text=text,
                                 org=(start_x, y),
                                 fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                 fontScale=0.55,
                                 color=color,
                                 thickness=2)

    return copy_frame

def display_loaded_cv2_images(data, indices=None, frame_name="Image"):
    if indices is not None:
        # Only fetch images at specified indices
        data = [data[i] for i in indices]

    for img in data:
        cv2.imshow(frame_name, img)
        cv2.waitKey(0)