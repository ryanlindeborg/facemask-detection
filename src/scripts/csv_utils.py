import csv
import cv2
import os

def create_labels_csv_from_dict(label_list_of_dict, csv_file_path):
    """Takes a list of dictionaries, and populates each csv row with one of these dictionaries, saving the csv file at the specified path"""

    with open(csv_file_path, mode="w") as csv_file:
        fieldnames = ["img_name", "label"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for label_dict in label_list_of_dict:
            writer.writerow(label_dict)

    print(f"Finished saving labels to csv file {csv_file_path}")

def read_images_and_labels_from_csv(csv_file_path, img_folder_path, img_extension):
    """Reads in the image names and labels from the csv for loading data in to evaluate the model"""

    data = []
    target = []
    with open(csv_file_path, mode="r") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            line_count += 1
            if line_count == 0:
                # First line is labels of csv
                continue
            img_name = row["img_name"]
            label = row["label"]

            img_path = os.path.join(img_folder_path, img_name + img_extension)
            img = cv2.imread(img_path)
            data.append(img)
            target.append(label)

    return data, target
