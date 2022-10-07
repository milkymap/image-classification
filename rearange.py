import click
import pandas as pd

from os import path 
from rich.progress import track 
from glob import glob


map_label2index = {
    'cat': 0,
    'dog': 1
}

map_label2reverse = {
    'cat': 'dog',
    'dog': 'cat'
}

@click.command()
@click.option('--path2images')
def process(path2images):
    image_paths = sorted(glob(path.join(path2images, '*.jpg')))
    acc = {
        'images': [],
        'cat': [],
        'dog': []
    }
    image_paths = image_paths[:128] + image_paths[-128:]
    for path_ in track(image_paths, 'label creation'):
        _, filename = path.split(path_)
        label = filename.split('.')[0]
        index = map_label2index[label]
        acc['images'].append(filename)
        acc[label].append(1)
        acc[map_label2reverse[label]].append(0)

    df = pd.DataFrame(data=acc)
    df.to_csv('label_config.csv', index=False)
        
        
if __name__ == '__main__':
    process()
