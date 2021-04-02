import numpy as np
import cv2


def detect_faces_locations_mtcnn(frame, mtcnn_model, confidence_thershold=0.5):
    """Detect faces and locations using mtcnn model"""

    # Make model prediction
    result = mtcnn_model.detect_faces(frame)

    # Initiliaze our list of faces and their corresponding locations and confidences
    faces = []
    locs = []
    confidences = []

    for person in result:

        bounding_box = person['box']
        confidence = person['confidence']

        if confidence > confidence_thershold:

            # Create x-y coordinates
            start_x = bounding_box[0]
            start_y = bounding_box[1]
            end_x = bounding_box[0] + bounding_box[2]
            end_y = bounding_box[1] + bounding_box[3]

            # Extract the ROI
            face = frame[start_y:end_y, start_x:end_x]

            # Add the face and bounding boxes to their respective lists
            faces.append(face)
            locs.append((start_x, start_y, end_x, end_y))
            confidences.append(confidence)

    return faces, locs, confidences


def detect_faces_locations_caffe(frame, face_net, confidence_threshold=0.5):
    """Detect faces and locations using caffe model"""

    # Construct a blob from a frame
    blob = cv2.dnn.blobFromImage(image=frame,
                                 scalefactor=1.0,
                                 size=(300, 300),
                                 mean=(104.0, 177.0, 123.0))

    # Pass the blob through the network and obtain the face detections
    face_net.setInput(blob)
    detections = face_net.forward()

    # Initiliaze our list of faces and their corresponding locations and confidences
    faces = []
    locs = []
    confidences = []

    # Grab the dimensions of the frame
    (h, w) = frame.shape[:2]

    # Loop over the detections
    for i in range(0, detections.shape[2]):

        # Extract the confidence associated with the detection
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:

            # Compute the (x, y) coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype("int")

            # Ensure the bounding boxes fall within the dimensions of the frame
            (start_x, start_y) = (max(0, start_x), max(0, start_y))
            (end_x, end_y) = (min(w - 1, end_x), min(h - 1, end_y))

            # Extract the ROI
            face = frame[start_y:end_y, start_x:end_x]

            # Add the face and bounding boxes to their respective lists
            faces.append(face)
            locs.append((start_x, start_y, end_x, end_y))
            confidences.append(confidence)

    return faces, locs, confidences


def draw_bounding_boxes(frame, locs, confidences):
    """Draw bounding boxes in image """

    # Keep original image intact
    copy_frame = np.copy(frame)

    for box, confidence in zip(locs, confidences):
        (start_x, start_y, end_x, end_y) = box

        # Format confidence from probability to percentage
        text = f"{confidence * 100:.2f}%"

        # Shift down 10 px y coordinate in case face detection occurs at the very top of the image
        y = start_y - 10 if start_y - 10 > 10 else start_y + 10

        # Draw rectangle in image
        copy_frame = cv2.rectangle(img=copy_frame,
                                   pt1=(start_x, start_y),
                                   pt2=(end_x, end_y),
                                   color=(0, 0, 255),
                                   thickness=2)

        # Put text with confidence in image
        copy_frame = cv2.putText(img=copy_frame,
                                 text=text,
                                 org=(start_x, y),
                                 fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                 fontScale=0.45,
                                 color=(0, 0, 255),
                                 thickness=2)

    return copy_frame


def detect_faces_and_locations(frame, model_name, face_net):
    """Detect faces and locations in an image"""

    if model_name == 'caffe':
        faces, locs, confidences = detect_faces_locations_caffe(frame, face_net)

    elif model_name == 'mtcnn':
        faces, locs, confidences = detect_faces_locations_mtcnn(frame, face_net)

    else:
        raise NotImplementedError(f'f{model_name} not available now')

    return faces, locs, confidences
