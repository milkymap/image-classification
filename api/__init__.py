import torch as th 

from os import getenv, path  
from flask import Flask 

from loguru import logger 
from libraries.strategies import load_vectorizer

PREDICTOR = getenv('PREDICTOR')
VECTORIZER = getenv('VECTORIZER')

print(PREDICTOR)
print(VECTORIZER)

current_path = path.dirname(__file__)
path2vectorizer = path.join(current_path, '..', VECTORIZER)
vectorizer = load_vectorizer(path2vectorizer)

logger.success('the vectorizer was loaded')

path2predictor = path.join(current_path, '..', PREDICTOR)
predictor = th.load(path2predictor)
predictor.eval()

logger.success('the predictor was loaded')

app = Flask(__name__)
app.config['labels'] = ['cat', 'dog']
app.config['predictor'] = predictor 
app.config['vectorizer'] = vectorizer 

from api import views 