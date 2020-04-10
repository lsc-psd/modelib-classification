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


def writeResultsInCsv(data_array):
    csvPath = 'test_data.csv'
    if not os.path.exists(csvPath):
        with open(csvPath, 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            for data in data_array:
                writer.writerow(data)
    else:
        with open(csvPath, 'a') as f:
            writer = csv.writer(f, lineterminator='\n')
            for data in data_array:
                writer.writerow(data)

def extension_check(filename):
    if imghdr.what(filename) == 'png':
        return print("image is png")
    elif imghdr.what(filename) == 'jpeg':
        return print("image is jpeg")
    else:
        print("Extension is different")
        sys.exit()

def test(read_default):

    test_path = read_default.get('test_dir')
    img_height = int(read_default.get('img_height'))
    img_width = int(read_default.get('img_width'))
    input_shape = (img_height, img_width, 3)
    n_categories = len(glob.glob(os.path.join(read_default.get('train_dir'), "*")))
    model_name = read_default.get('model_name')
    weigh_path = read_default.get('weight_path')

    extension_check(test_path)
    Structure = import_module(f'models.{model_name}')
    model = Structure.build(input_shape, n_categories)
    weight = glob.glob(weigh_path)
    model.load_weights(weight)

    image = Image.open(test_path).resize((img_height, img_width))
    image = np.array(image)
    prediction = model.predict([[image]])
    name = os.path.basename(test_path)
    print(str(name) + " : " + str(prediction))
    data = [name, prediction]
    writeResultsInCsv(data)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', default='config.ini', type=str, help='config file')
    args = parser.parse_args()

    config_ini = configparser.ConfigParser()
    config_ini.read(args.c, encoding='utf-8')
    read_default = config_ini['MODELIB']

    test(read_default)