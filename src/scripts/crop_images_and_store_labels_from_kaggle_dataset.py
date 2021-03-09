import os
from PIL import Image
import xml.etree.ElementTree as ET

# Ubuntu box
# KAGGLE_IMAGES_FILEPATH = "/home/ryanlindeborg/projects/ml/facemask/data/kaggle-facemask-data/images"
# KAGGLE_LABELS_FILEPATH = "/home/ryanlindeborg/projects/ml/facemask/data/kaggle-facemask-data/annotations"
# CROPPED_IMG_FOLDER = "/home/ryanlindeborg/projects/ml/facemask/data/kaggle-facemask-data/cropped_images"
# Mac local
KAGGLE_IMAGES_FILEPATH = "/Users/ryanlindeborg/Desktop/dev/ml-learning/projects/facemask/kaggle-facemask-data/images"
KAGGLE_LABELS_FILEPATH = "/Users/ryanlindeborg/Desktop/dev/ml-learning/projects/facemask/kaggle-facemask-data/annotations"
CROPPED_IMG_FOLDER = "/Users/ryanlindeborg/Desktop/dev/ml-learning/projects/facemask/kaggle-facemask-data/cropped_images"

XML_FILE_EXTENSION = ".xml"
PNG_FILE_EXTENSION = ".png"

STATUS_WITH_MASK = 1
STATUS_WITHOUT_MASK = 0

def crop_images_and_store_labels_from_kaggle_dataset(img_folder, annotation_folder, cropped_img_folder):
    for img_file in os.listdir(img_folder):
        img_path = os.path.join(img_folder, img_file)
        original_img = Image.open(img_path)
        # Image label = image file name without extension
        image_label = os.path.splitext(img_file)[0]
        xml_file = image_label + XML_FILE_EXTENSION
        xml_path = os.path.join(annotation_folder, xml_file)

        root = ET.parse(xml_path).getroot()
        object_count = 0
        for object in root.findall('object'):
            object_count += 1
            label = object.find('name').text
            label = STATUS_WITH_MASK if label == "with_mask" else STATUS_WITHOUT_MASK
            bbox = object.find('bndbox')
            x1 = int(bbox.find('xmin').text)
            x2 = int(bbox.find('xmax').text)
            y1 = int(bbox.find('ymin').text)
            y2 = int(bbox.find('ymax').text)

            # Crop image of face from larger image, and store label
            cropped_img = original_img.crop((x1, y1, x2, y2))
            cropped_img_path = os.path.join(cropped_img_folder, image_label + '_obj' + str(object_count) + '_label' + str(label) + PNG_FILE_EXTENSION)
            # cropped_img.show()
            cropped_img.save(cropped_img_path)


        # Temp break for debugging
        # break


crop_images_and_store_labels_from_kaggle_dataset(KAGGLE_IMAGES_FILEPATH, KAGGLE_LABELS_FILEPATH, CROPPED_IMG_FOLDER)