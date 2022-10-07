import cv2 
import pickle, json 

import numpy as np 
import operator as op 
import itertools as it, functools as ft 

import torch as th 
import torch.nn as nn 
import torch.nn.functional as F 

from os import path 
from glob import glob 
from torchvision import models 

def serialize(data, path2dump):
    with open(path2dump, mode='wb') as fp:
        pickle.dump(data, fp)

def deserialize(path2dump):
    with open(path2dump, mode='rb') as fp:
        data = pickle.load(fp)
    return data 

def cv2th(image):
    blue, green, red = cv2.split(image)
    return th.as_tensor(np.stack([red, green, blue])).float()

def read_image(path2image, size=None):
    image = cv2.imread(path2image, cv2.IMREAD_COLOR)
    if size is not None:
        image = cv2.resize(image, size)
    return image 

def pull_files(path2files, extension):
    real_path = path.join(path2files, f'*.{extension}')
    file_paths = sorted(glob(real_path))
    return file_paths 

def load_vectorizer(path2vectorizer):
    if not path.isfile(path2vectorizer):
        vectorizer = models.resnet18(pretrained=True, progress=True)
        for prm in vectorizer.parameters():
            prm.requires_grad = False 
        vectorizer.eval()
        vectorizer = nn.Sequential(*list(vectorizer.children())[:-1])
        th.save(vectorizer, path2vectorizer)
    else:
        vectorizer = th.load(path2vectorizer)
    return vectorizer

