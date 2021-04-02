from tensorflow.keras.models import load_model
from utils import prepare_data_to_model, read_images_and_image_paths_from_data_folder
import numpy as np

WRONG_PREDICTIONS_CSV_FILE_PATH = "./wrong_predictions_img_paths.csv"

# Prepare data to ingest model
data, target, img_paths = read_images_and_image_paths_from_data_folder(data_path='./data/validation_data')
data, target = prepare_data_to_model(data, target)

# Build a neural network
model = load_model('./models/final/mask_model.h5')

# Predict on the test set, and compare to true labels to record the examples which our model classified incorrectly
prediction_classes = model.predict_classes(data)
# Finding the index of the max arg (1) will give us the target label
target_values = np.argmax(target, axis=1)
wrong_prediction_indices = [i for i, prediction_class in enumerate(prediction_classes) if prediction_class != target_values[i]]
wrong_prediction_img_paths = img_paths[wrong_prediction_indices]

# Write wrong prediction image paths numpy array to a csv
np.savetxt(WRONG_PREDICTIONS_CSV_FILE_PATH, wrong_prediction_img_paths, delimiter=",")