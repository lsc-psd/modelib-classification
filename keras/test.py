import configparser
import csv
import glob
import os
from importlib import import_module

import numpy as np
from PIL import Image


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

def test(read_default):

    test_path = read_default.get('test_dir')
    img_height = int(read_default.get('img_height'))
    img_width = int(read_default.get('img_width'))
    input_shape = (img_height, img_width, 3)
    n_categories = int(read_default.get('n_categories'))
    model_name = read_default.get('model_name')
    weigh_path = read_default.get('weight_path')

    Structure = import_module(f'models.{model_name}')
    model = Structure.build(input_shape, n_categories)
    weight = glob.glob(weigh_path)
    model.load_weights(weight)

    image_path = glob.glob(os.path.join(test_path, "*"))
    data = []
    for path in image_path:
        image = Image.open(path).resize((img_height, img_width), Image.LANCZOS)
        image = np.array(image)
        prediction = model.predict([[image]])
        name = os.path.basename(path)
        data.append([name, prediction])
        print(str(name) + " : " + str(prediction))
    writeResultsInCsv(data)
    
if __name__ == '__main__':

    config_ini = configparser.ConfigParser()
    config_ini.read('config.ini', encoding='utf-8')
    read_default = config_ini['DEFAULT']

    test(read_default)