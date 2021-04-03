from tensorflow.keras.models import load_model
from utils import prepare_data_to_model, read_images_and_image_paths_from_data_folder
import numpy as np

WRONG_PREDICTIONS_CSV_FILE_PATH = "./wrong_predictions_img_paths.csv"
CORRECT_PREDICTIONS_CSV_FILE_PATH = "./correct_predictions_img_paths.csv"
DATA_FILE_PATH = "./data/validation_data"
MODEL_FILE_PATH = "./models/final/mask_model.h5"

def get_examples_from_prediction(misclasssifed=True, data_file_path=DATA_FILE_PATH, model_file_path=MODEL_FILE_PATH, predictions_csv_file_path=WRONG_PREDICTIONS_CSV_FILE_PATH):
    # Prepare data to ingest model
    data, target, img_paths = read_images_and_image_paths_from_data_folder(data_path=data_file_path)
    data, target = prepare_data_to_model(data, target)

    # Build a neural network
    model = load_model(model_file_path)

    # Predict on the test set, and compare to true labels to record the examples which our model classified incorrectly
    prediction_classes = model.predict_classes(data)
    # Finding the index of the max arg (1) will give us the target label
    target_values = np.argmax(target, axis=1)
    if misclasssifed:
        # Get list of misclassified examples
        prediction_indices = [i for i, prediction_class in enumerate(prediction_classes) if prediction_class != target_values[i]]
    else:
        # Get list of correctly predicted examples
        prediction_indices = [i for i, prediction_class in enumerate(prediction_classes) if prediction_class == target_values[i]]

    prediction_img_paths = img_paths[prediction_indices]

    # Write wrong prediction image paths numpy array to a csv
    np.savetxt(predictions_csv_file_path, prediction_img_paths, delimiter=",", fmt="%s")

if __name__ == "__main__":
    # Fetch incorrectly predicted image list
    get_examples_from_prediction(misclasssifed=True, predictions_csv_file_path=WRONG_PREDICTIONS_CSV_FILE_PATH)
    # Fetch correctly predicted image list
    get_examples_from_prediction(misclasssifed=False, predictions_csv_file_path=CORRECT_PREDICTIONS_CSV_FILE_PATH)