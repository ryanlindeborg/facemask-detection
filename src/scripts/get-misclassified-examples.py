from tensorflow.keras.models import load_model
from utils import prepare_data_to_model, read_images_from_data_folder

# Prepare data to ingest model
data, target = read_images_from_data_folder(data_path='./data/validation_data')
data, target = prepare_data_to_model(data, target)

# Build a neural network
model = load_model('./models/final/mask_model.h5')

# Predict on the test set, and compare to true labels to record the examples which our model classified incorrectly
predictions = model.predict(data)
prediction_classes = model.predict_classes(data)
wrong_predictions = data[predictions != target]

# indices = [i for i,v in enumerate(pred) if pred[i]!=y_test[i]]
# subset_of_wrongly_predicted = [x_test[i] for i in indices ]

# indices = [i for i in range(len(y_test)) if y_test[i] != y_pred[i]]