from utils import preprocess_training_data


def detect_mask(frame, model, img_size):
    """Return probability of having a mask"""

    frame = preprocess_training_data(frame, img_size)
    predictions = model.predict(frame)

    return predictions
