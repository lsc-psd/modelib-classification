import configparser
import sys
import csv
import glob
import os
from importlib import import_module
import numpy as np
from PIL import Image
import imghdr
import argparse


def writeResultsInCsv(categories, data):
    csvPath = 'test_data.csv'
    if not os.path.exists(csvPath):
        with open(csvPath, 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(["", categories])
            writer.writerow(data)
    else:
        with open(csvPath, 'a') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(data)

def extension_check(filename):
    for name in filename:
        if imghdr.what(name) == 'png':
            return print("image is png")
        elif imghdr.what(name) == 'jpeg':
            return print("image is jpeg")
        elif imghdr.what(name) == 'jpg':
            return print("image is jpg")
        else:
            print("Extension is different")
            sys.exit()

def test(read_default):

    test_path = glob.glob(os.path.join(read_default.get('test_dir'), "*"))
    img_height = int(read_default.get('img_height'))
    img_width = int(read_default.get('img_width'))
    input_shape = (img_height, img_width, 3)
    categories = glob.glob((os.path.join(read_default.get('train_dir'), "*")))
    print(categories)
    model_name = read_default.get('model_name')
    weigh_path = glob.glob(os.path.join(read_default.get('weight_path'), "*"))

    extension_check(test_path)

    Structure = import_module(f'models.{model_name}')
    model = Structure.build(input_shape, len(categories))
    model.load_weights(weigh_path[0])

    category_name_list = []
    for category_name in categories:
        category_name_list.append(os.path.basename(category_name))

    for path in test_path:
        image = Image.open(path).resize((img_height, img_width))
        image = np.array(image)
        prediction = model.predict([[image]])
        name = os.path.basename(path)
        print(str(name) + " : " + str(prediction))
        data = [name, prediction]
        writeResultsInCsv(category_name_list, data)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', default='config.ini', type=str, help='config file')
    args = parser.parse_args()
    config_ini = configparser.ConfigParser()
    config_ini.read(args.c, encoding='utf-8')
    read_default = config_ini['MODELIB']

    test(read_default)