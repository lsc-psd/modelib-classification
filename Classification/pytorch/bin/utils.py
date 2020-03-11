import pdb
import pickle


def safe_run(func):
    def func_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(e)
            self = args[0]
            pdb.set_trace()
            return None
    return func_wrapper


def config_save(args):
    with open('./last_config.pickle', 'w') as f:
        pickle.dump(args, f)


def config_load(path):
    with open(path, 'r') as f:
        config = pickle.load(f)
    return config