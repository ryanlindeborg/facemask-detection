import tensorflow as tf
import os

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

from utils import create_model, prepare_data_to_model, read_images_from_data_folder

# Set random seed
SEED = 1
tf.random.set_seed(SEED)

# Deterministic run on gpu
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Prepare data to ingest model
data, target = read_images_from_data_folder(data_path='./data/')
data, target = prepare_data_to_model(data, target)

# Build a neural network
model = create_model(input_shape=data.shape[1:])

# Split data in train and test
train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.15, random_state=SEED)

# Train model and save checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='min', restore_best_weights=True)
checkpoint = ModelCheckpoint('./models/checkpoints/model-{epoch:03d}.model', monitor='val_loss', verbose=1,
                             save_best_only=True, mode='auto')
history = model.fit(train_data, train_target, epochs=20, callbacks=[checkpoint, early_stopping], validation_split=0.2)

# Evaluate model on test data
print(model.evaluate(test_data, test_target))

# Save model
model.save('./models/final/mask_model.h5')
