from utils import preprocess_training_data


def detect_mask(face_frames, model, img_size):
    """Return probability of having a mask"""

    face_frames = preprocess_training_data(face_frames, img_size)
    predictions = model.predict(face_frames)

    return predictions
