from PIL import Image
from importlib import import_module
import csv


def writeResultsInCsv(name, prediction):

    appendData = [name, prediction]
    csvPath = name + '.csv'
    if not os.path.exists(csvPath):
        with open(csvPath, 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(appendData)
    else:
        with open(csvPath, 'a') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(appendData)

def test(read_default):

    image_path = read_default.get('test_dir')
    img_height = int(read_default.get('img_height'))
    img_width = int(read_default.get('img_width'))
    input_shape = (img_height, img_width, 3)
    n_categories = int(read_default.get('n_categories'))
    model_name = read_default.get('model_name')
    weigth_path = read_default.get('weight_path')

    image = Image.open(image_path).resize((img_height, img_width), Image.LANCZOS)
    image = np.array(image)

    Structure = import_module(f'models.{model_name}')
    model = Structure.build(input_shape, n_categories)
    model.load_weights(weigth_path)
    prediction = model.predict(image)

    name = os.path.basename(image_path)
    writeResultsInCsv(name, prediction)
    print(name, prediction)
    
if __name__ == '__main__':

    config_ini = configparser.ConfigParser()
    config_ini.read('config.ini', encoding='utf-8')
    read_default = config_ini['DEFAULT']

    test(read_default)
