from tensorflow.keras.models import load_model
from utils import create_model, prepare_data_to_model, read_images_from_data_folder, fit_model

# Prepare data to ingest model
data, target = read_images_from_data_folder(data_path='./data/validation-data')
data, target = prepare_data_to_model(data, target)

# Build a neural network
model = load_model('./models/final/mask_model.h5')

# Evaluate model on test data
val_loss, val_acc = model.evaluate(data, target)
print("Final model results:")
print(f"Val loss: {val_loss}")
print(f"Val acc: {val_acc}")
