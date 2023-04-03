import distracted
import pandas as pd
from PIL import Image
from pathlib import Path
from enum import Enum

DATA_PATH = Path(distracted.__path__[0]).parents[1] / "data"
assert DATA_PATH.exists()

def get_train_df():
    classes = {
    'c0': 'safe driving',
    'c1': 'texting - right',
    'c2': 'talking on the phone - right',
    'c3': 'texting - left',
    'c4': 'talking on the phone - left',
    'c5': 'operating the radio',
    'c6': 'drinking',
    'c7': 'reaching behind',
    'c8': 'hair and makeup',
    'c9': 'talking to passenger',
    }
    name_to_path = {f.name:f for f in DATA_PATH.rglob('*.jpg')}
    # test_files = [f for f in (DATA_PATH / 'imgs/test').rglob('*.jpg')]

    df = pd.read_csv(DATA_PATH / 'driver_imgs_list.csv').assign(path=df.img.map(name_to_path), desc=df.classname.map(classes))
    return df

class ImgLoader:
    def __init__(self, source: Source):


    def iter_images(self) -> Path:
        yield from ()
    def iter_images(self) -> Image.Image:
        yield from 
    image = DATA_PATH / "imgs/train/c1/img_34.jpg"
