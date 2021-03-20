from tensorflow.keras.models import load_model
from utils import create_model, prepare_data_to_model, read_images_from_data_folder, fit_model

# Prepare data to ingest model
data, target = read_images_from_data_folder(data_path='./data/validation-data')
data, target = prepare_data_to_model(data, target)

# Build a neural network
model = load_model('./models/final/mask_model.h5')

# Evaluate model on test data
print(f"Final model accuracy: {model.evaluate(data, target)}")
