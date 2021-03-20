import os
from PIL import Image
import xml.etree.ElementTree as ET
from src.scripts.csv_utils import create_labels_csv_from_dict

# Ubuntu box
KAGGLE_IMAGES_FILEPATH = "/home/ryanlindeborg/projects/ml/facemask/data/kaggle-facemask-data/images"
KAGGLE_LABELS_FILEPATH = "/home/ryanlindeborg/projects/ml/facemask/data/kaggle-facemask-data/annotations"
CROPPED_IMG_FOLDER = "/home/ryanlindeborg/projects/ml/facemask/data/kaggle-facemask-data/cropped_images"
OUTPUT_LABELS_CSV_FILE_PATH = "/home/ryanlindeborg/projects/ml/facemask/code/output_labels.csv"
# Mac local
# KAGGLE_IMAGES_FILEPATH = "/Users/ryanlindeborg/Desktop/dev/ml-learning/projects/facemask/kaggle-facemask-data/images"
# KAGGLE_LABELS_FILEPATH = "/Users/ryanlindeborg/Desktop/dev/ml-learning/projects/facemask/kaggle-facemask-data/annotations"
# CROPPED_IMG_FOLDER = "/Users/ryanlindeborg/Desktop/dev/ml-learning/projects/facemask/kaggle-facemask-data/cropped_images"
# OUTPUT_LABELS_CSV_FILE_PATH = "/Users/ryanlindeborg/Desktop/dev/ml-learning/projects/facemask/kaggle-facemask-data/output_labels.csv"

XML_FILE_EXTENSION = ".xml"
PNG_FILE_EXTENSION = ".png"

LABEL_WITH_MASK = 1
LABEL_WITHOUT_MASK = 0

def crop_images_and_store_labels_from_kaggle_dataset(img_folder, annotation_folder, cropped_img_folder):
    # Will return dictionary with new image names and corresponding label
    img_label_list_of_dict = []
    for img_file in os.listdir(img_folder):
        img_path = os.path.join(img_folder, img_file)
        original_img = Image.open(img_path)
        # Image label = image file name without extension
        image_name = os.path.splitext(img_file)[0]
        xml_file = image_name + XML_FILE_EXTENSION
        xml_path = os.path.join(annotation_folder, xml_file)

        root = ET.parse(xml_path).getroot()
        object_count = 0
        for object in root.findall('object'):
            object_count += 1
            label = object.find('name').text
            label = LABEL_WITH_MASK if label == "with_mask" else LABEL_WITHOUT_MASK
            bbox = object.find("bndbox")
            x1 = int(bbox.find("xmin").text)
            x2 = int(bbox.find("xmax").text)
            y1 = int(bbox.find("ymin").text)
            y2 = int(bbox.find("ymax").text)

            # Crop image of face from larger image, and store label
            cropped_img = original_img.crop((x1, y1, x2, y2))
            cropped_img_name = image_name + "_obj" + str(object_count) + "_label" + str(label)
            cropped_img_path = os.path.join(cropped_img_folder, cropped_img_name + PNG_FILE_EXTENSION)
            # cropped_img.show()
            cropped_img.save(cropped_img_path)

            # Populate label dictionary
            current_label_dict = {}
            current_label_dict["img_name"] = cropped_img_name
            current_label_dict["label"] = label
            img_label_list_of_dict.append(current_label_dict)

    return img_label_list_of_dict

if __name__ == "__main__":
    img_label_dict = crop_images_and_store_labels_from_kaggle_dataset(KAGGLE_IMAGES_FILEPATH, KAGGLE_LABELS_FILEPATH, CROPPED_IMG_FOLDER)
    create_labels_csv_from_dict(label_list_of_dict=img_label_dict, csv_file_path=OUTPUT_LABELS_CSV_FILE_PATH)