import logging
import cv2

from tensorflow.keras.models import load_model
from utils import draw_bounding_boxes_and_confidences

from face_detection import detect_faces_and_locations
from mask_detection import detect_mask
from tests.test_face_detection import display_video

logger = logging.getLogger(__name__)


def load_facemask_model(path):
    """Load trained facemask detection model"""
    facemask_model = load_model(path)

    return facemask_model


def detect_facemask_img(image_path, face_model_name, facemask_model_path):

    logger.info(f"Starting facemask detection in image")
    facemask_model = load_facemask_model(facemask_model_path)
    image = cv2.imread(image_path)
    faces, locs, confidences = detect_faces_and_locations(image, face_model_name)
    preds = detect_mask(faces, facemask_model, 150)
    mask_probabilities = [pred[1] for pred in preds]
    image = draw_bounding_boxes_and_confidences(image, locs, mask_probabilities)
    cv2.imshow("Face Mask detection", image)
    cv2.waitKey(0)


def detect_facemask_video(face_model_name, facemask_model_path):

    logger.info(f"Starting webcam  detection in image")
    video_source = cv2.VideoCapture(0)

    while True:

        _, frame = video_source.read()
        facemask_model = load_facemask_model(facemask_model_path)
        faces, locs, confidences = detect_faces_and_locations(frame, face_model_name)
        preds = detect_mask(faces, facemask_model, 150)
        mask_probabilities = [pred[1] for pred in preds]
        frame = draw_bounding_boxes_and_confidences(frame, locs, mask_probabilities)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def main():
    logging.basicConfig(format='%(asctime)s :: %(levelname)s :: %(name)s  :: %(message)s', level=logging.INFO)
    detect_facemask_img('./data/face_detection_data/mask_vigo.jpeg',
                        face_model_name='caffe',
                        facemask_model_path='./models/final/mask_model.h5')
    detect_facemask_video(face_model_name='caffe', facemask_model_path='./models/final/mask_model.h5')


if __name__ == "__main__":
    main()
