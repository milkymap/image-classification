import cv2 
import json 

import torch as th 
import numpy as np 

from libraries.strategies import cv2th 
from flask import request
from api import app 

@app.route('/')
def index():
    return json.dumps({
        'status': 1, 
        'message': 'server is up...!'
    })

@app.route('/predict', methods=['POST'])
def prediction():
    # read data 
    file_handler = request.files['data']
    binary_stream = file_handler.read()  
    image = cv2.imdecode(np.frombuffer(binary_stream, np.uint8), cv2.IMREAD_COLOR)
    print(image)

    # prepare data 
    tensor = cv2th(image)
    input_batch = tensor[None, ...]
    fingerprint = th.flatten(app.config['vectorizer'](input_batch))

    # make prediction 
    with th.no_grad():
        probabilities = app.config['predictor'](fingerprint[None, ...])
        probabilities = th.softmax(probabilities, dim=1).squeeze(0)
        index = th.argmax(probabilities)
        label = app.config['labels'][index]

    # send response to client 
    return json.dumps({
        'status': 1,
        'message': 'success',
        'contents': f'class => a {label} was detected'
    })
