from tensorflow.saved_model import load as tfLoad
from os.path import exists

def getModel(_path: str):
    if not exists(_path):
        raise FileNotFoundError(f"{_path} does not exists")

    model = tfLoad(_path)
    return model


if  __name__ == "__main__":
    model = getModel('../tf/model')