import argparse


def main(args):
    
    model = System()
    model = model.load_model()

    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', default='./last_config.pickle', type=str, help='config file for models')
    parser.add_argument('-ckpt', type=str, help='checkpoint')
    parser.add_argument('-tags_csv', type=str, help='tags_csv file')
    parser.add_argument('-f', default='test_imgs', type=str, help='data folder path')
    args = parser.parse_args()
    main(args)
