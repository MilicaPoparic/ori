
import shutil
import os
import csv

def read_data():
    result = {}
    with open("chest_xray_data_set/metadata/chest_xray_metadata.csv", 'r') as file:
        csv_file = csv.DictReader(file)
        for row in csv_file:
            key = row.pop('X_ray_image_name')
            # print(key)
            result[key] = row
    # print(result['IM-0128-0001.jpeg'].get('Label')) ovo je da li je pneumonia ili normal
    return result

def classify_data(result):
    # u listama se nalaze imena fajlova
    bacteria = []
    normal = []
    virus = []
    if not os.path.exists('classified'):
        os.makedirs('classified')
    if not os.path.exists('classified/normal'):
        os.makedirs('classified/normal')
    if not os.path.exists('classified/virus'):
        os.makedirs('classified/virus')
    if not os.path.exists('classified/bacteria'):
        os.makedirs('classified/bacteria')

    for file in os.listdir('chest_xray_data_set'):
        if not os.path.isfile('chest_xray_data_set/' + file):
            continue
        # proveri kakva su pluca
        if not result.keys().__contains__(file):
            continue

        if result[file].get('Label') == "Normal":
            normal.append(file)
        elif result[file].get('Label_1_Virus_category') == "bacteria":
            bacteria.append(file)
        elif result[file].get('Label_1_Virus_category') == "Virus":
            virus.append(file)

    for f in normal:
        source_folder = "chest_xray_data_set/" + f
        dest_folder = "classified/normal"
        shutil.move(source_folder, dest_folder)
    for f in bacteria:
        source_folder = "chest_xray_data_set/" + f
        dest_folder = "classified/bacteria"
        shutil.move(source_folder, dest_folder)
    for f in virus:
        source_folder = "chest_xray_data_set/" + f
        dest_folder = "classified/virus"
        shutil.move(source_folder, dest_folder)

def read_test_data():
    result = {}
    with open("chest-xray-dataset-test/chest_xray_test_dataset.csv", 'r') as file:
        csv_file = csv.DictReader(file)
        for row in csv_file:
            key = row.pop('X_ray_image_name')
            result[key] = row
    classify_test_data(result)

def classify_test_data(result):
    bacteria = []
    normal = []
    virus = []
    if not os.path.exists('test'):
        os.makedirs('test')
    if not os.path.exists('test/normal'):
        os.makedirs('test/normal')
    if not os.path.exists('test/virus'):
        os.makedirs('test/virus')
    if not os.path.exists('test/bacteria'):
        os.makedirs('test/bacteria')

    for file in os.listdir('chest-xray-dataset-test/test'):
        if not os.path.isfile('chest-xray-dataset-test/test/' + file):
            continue
        # proveri kakva su pluca
        if not result.keys().__contains__(file):
            continue

        if result[file].get('Label') == "Normal":
            normal.append(file)
        elif result[file].get('Label_1_Virus_category') == "bacteria":
            bacteria.append(file)
        elif result[file].get('Label_1_Virus_category') == "Virus":
            virus.append(file)

    for f in normal:
        source_folder = "chest-xray-dataset-test/test/" + f
        dest_folder = "test/normal"
        shutil.move(source_folder, dest_folder)
    for f in bacteria:
        source_folder = "chest-xray-dataset-test/test/" + f
        dest_folder = "test/bacteria"
        shutil.move(source_folder, dest_folder)
    for f in virus:
        source_folder = "chest-xray-dataset-test/test/" + f
        dest_folder = "test/virus"
        shutil.move(source_folder, dest_folder)
