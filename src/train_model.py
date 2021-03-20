import tensorflow as tf
import os

from sklearn.model_selection import train_test_split

from utils import create_model, prepare_data_to_model, read_images_from_data_folder, fit_model

# Set random seed
SEED = 1
tf.random.set_seed(SEED)

# Deterministic run on gpu
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Prepare data to ingest model
data, target = read_images_from_data_folder(data_path='./data/training_data')
data, target = prepare_data_to_model(data, target)

# Build a neural network
model = create_model(input_shape=data.shape[1:])

# Split data in train and test
train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.15, random_state=SEED)

# Train model
model, history = fit_model(train_data, train_target, model)

# Evaluate model on test data
print(model.evaluate(test_data, test_target))

# Save model
model.save('./models/final/mask_model.h5')
