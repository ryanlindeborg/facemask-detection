import cv2
import logging

from face_detection import detect_faces_and_locations, draw_bounding_boxes, load_face_model

logger = logging.getLogger(__name__)


def test_face_detection_img(img_path, model_name, show_individual_faces=False):
    """Test face detection model in images"""

    logger.info(f"Starting face detection in image with model {model_name}")
    image = cv2.imread(img_path)
    faces, locs, confidences = detect_faces_and_locations(image, model_name)
    image = draw_bounding_boxes(image, locs, confidences)
    cv2.imshow("Image", image)

    if show_individual_faces:
        for i, face in enumerate(faces):
            cv2.imshow(f"Face {i}", face)

    cv2.waitKey(0)


def test_face_detection_webcam(model_name):
    """Test face detection model in real time with webcam"""

    logger.info(f"Starting face detection in webcam with model {model_name}")
    video_capture = cv2.VideoCapture(0)

    while True:
        _, frame = video_capture.read()
        faces, locs, confidences = detect_faces_and_locations(frame, model_name)
        frame = draw_bounding_boxes(frame, locs, confidences)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    video_capture.release()


def main():
    logging.basicConfig(format='%(asctime)s :: %(levelname)s :: %(name)s  :: %(message)s', level=logging.INFO)
    test_face_detection_webcam(model_name='caffe')
    test_face_detection_img('./data/face_detection_data/group_4.jpg', model_name='mtcnn')


if __name__ == "__main__":
    main()
