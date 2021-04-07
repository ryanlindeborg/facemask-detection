import cv2
import logging

from face_detection import detect_faces_and_locations, load_face_model
from utils import draw_bounding_boxes_and_confidences
# from imutils.video import FileVideoStream

logger = logging.getLogger(__name__)


def display_video(video_source, face_net, model_type):
    """Display video with opencv"""

    ret, frame = video_source.read()

    if ret:
        faces, locs, confidences = detect_faces_and_locations(frame, face_net, model_type)
        frame = draw_bounding_boxes_and_confidences(frame, locs, confidences)
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF
    else:
        key = ord("q")

    return key


def test_face_detection_img(img_path, model_type, show_individual_faces=False):
    """Test face detection model in images"""

    logger.info(f"Starting face detection in image with model {model_type}")
    image = cv2.imread(img_path)
    face_net = load_face_model(model_type)
    faces, locs, confidences = detect_faces_and_locations(image, face_net, model_type)
    image = draw_bounding_boxes_and_confidences(image, locs, confidences, False)
    cv2.imshow("Image", image)

    if show_individual_faces:
        for i, face in enumerate(faces):
            cv2.imshow(f"Face {i}", face)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_face_detection_video(model_type, source_type='webcam', video_path=None):
    """Test face detection model in real time with webcam or from video"""

    logger.info(f"Starting face detection in webcam with model {model_type}")

    face_net = load_face_model(model_type)
    source = 0 if source_type == 'webcam' else video_path
    video_source = cv2.VideoCapture(source)

    if source_type == 'webcam':

        while True:
            key = display_video(video_source, face_net, model_type)
            if key == ord("q"):
                break

    else:

        # video_source = FileVideoStream(source)
        while video_source.isOpened():
            key = display_video(video_source, face_net, model_type)
            if key == ord("q"):
                break

    video_source.release()
    cv2.destroyAllWindows()


def main():
    logging.basicConfig(format='%(asctime)s :: %(levelname)s :: %(name)s  :: %(message)s', level=logging.INFO)
    test_face_detection_video(model_type='caffe', source_type='webcam')
    test_face_detection_video(model_type='caffe', source_type='video',
                              video_path='./data/face_detection_data/ig_pollosus_mask_3.mp4')
    test_face_detection_img('./data/face_detection_data/group_4.jpg', model_type='caffe')
    test_face_detection_img('./data/face_detection_data/group_3.jpg', model_type='mtcnn')


if __name__ == "__main__":
    main()
