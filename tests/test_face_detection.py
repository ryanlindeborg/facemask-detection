import cv2
import logging

from face_detection import detect_faces_and_locations, draw_bounding_boxes
from mtcnn.mtcnn import MTCNN


logger = logging.getLogger(__name__)


def load_model(model_name: str):
    """
    Load our serialized face detector model from disk

    :param model_name: caffe or mtcnn
    :return: trained face detection model
    """

    if model_name == 'caffe':
        prototxt_path = "./face_clasifiers/deploy.prototxt"
        weights_path = "./face_clasifiers/res10_300x300_ssd_iter_140000.caffemodel"
        face_net = cv2.dnn.readNet(prototxt_path, weights_path)

    elif model_name == 'mtcnn':
        face_net = MTCNN()

    else:
        raise NotImplementedError(f'f{model_name} not available now')

    return face_net


def test_face_detection_img(img_path, model_name):
    """Test face detection model in images"""

    logger.info(f"Starting face detection in image with model {model_name}")
    image = cv2.imread(img_path)
    face_net = load_model(model_name)
    faces, locs, confidences = detect_faces_and_locations(image, model_name, face_net)
    image = draw_bounding_boxes(image, locs, confidences)
    cv2.imshow("Image", image)
    cv2.waitKey(0)


def test_face_detection_webcam(model_name):
    """Test face detection model in real time with webcam"""

    logger.info(f"Starting face detection in webcam with model {model_name}")
    video_capture = cv2.VideoCapture(0)
    face_net = load_model(model_name)

    while True:
        _, frame = video_capture.read()
        faces, locs, confidences = detect_faces_and_locations(frame, model_name, face_net)
        frame = draw_bounding_boxes(frame, locs, confidences)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    video_capture.release()


def main():
    logging.basicConfig(format='%(asctime)s :: %(levelname)s :: %(name)s  :: %(message)s', level=logging.INFO)
    test_face_detection_webcam(model_name='caffe')
    test_face_detection_img('./data/face_detection_data/group_1.jpg', model_name='caffe')


if __name__ == "__main__":
    main()
