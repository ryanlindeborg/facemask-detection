import cv2
import logging

from face_detection import detect_faces_and_locations
from utils import draw_bounding_boxes_and_confidences

logger = logging.getLogger(__name__)


def display_video(video_source, model_name):
    """Display video with opencv"""

    ret, frame = video_source.read()

    if ret:
        faces, locs, confidences = detect_faces_and_locations(frame, model_name)
        frame = draw_bounding_boxes_and_confidences(frame, locs, confidences)
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF
    else:
        key = ord("q")

    return key


def test_face_detection_img(img_path, model_name, show_individual_faces=False):
    """Test face detection model in images"""

    logger.info(f"Starting face detection in image with model {model_name}")
    image = cv2.imread(img_path)
    faces, locs, confidences = detect_faces_and_locations(image, model_name)
    image = draw_bounding_boxes_and_confidences(image, locs, confidences, False)
    cv2.imshow("Image", image)

    if show_individual_faces:
        for i, face in enumerate(faces):
            cv2.imshow(f"Face {i}", face)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_face_detection_video(model_name, source_type='webcam', video_path=None):
    """Test face detection model in real time with webcam or from video"""

    logger.info(f"Starting face detection in webcam with model {model_name}")

    source = 0 if source_type == 'webcam' else video_path
    video_source = cv2.VideoCapture(source)

    if source_type == 'webcam':

        while True:
            key = display_video(video_source, model_name)
            if key == ord("q"):
                break

    else:

        while video_source.isOpened():
            key = display_video(video_source, model_name)
            if key == ord("q"):
                break

    video_source.release()
    cv2.destroyAllWindows()


def main():
    logging.basicConfig(format='%(asctime)s :: %(levelname)s :: %(name)s  :: %(message)s', level=logging.INFO)
    test_face_detection_video(model_name='caffe', source_type='video',
                              video_path='./data/face_detection_data/ig_pollosus_mask_1.mp4')
    test_face_detection_img('./data/face_detection_data/group_4.jpg', model_name='caffe')
    test_face_detection_img('./data/face_detection_data/group_3.jpg', model_name='mtcnn')
    test_face_detection_video(model_name='caffe', source_type='webcam')


if __name__ == "__main__":
    main()
