import cv2 
import click
import requests

from loguru import logger
from torch import Tensor 
from libraries.strategies import *
from rich.progress import track

from neuralnet import NeuralNet
from torch.utils.data import TensorDataset, DataLoader 

@click.group(chain=False, invoke_without_command=True)
@click.pass_context
def router_command(ctx):
    pass 

@router_command.command()
@click.option('--path2images')
@click.option('--path2vectorizer')
def processing(path2images, path2vectorizer):
    map_class2index = {
        'cat': 0,
        'dog': 1
    }
    vectorizer = load_vectorizer(path2vectorizer)
    logger.success('the vectorizer was loaded')

    file_paths = pull_files(path2images, 'jpg')
    logger.debug(f'nb images : {len(file_paths):05d}')
    logger.debug('extraction will start')

    features_accumulator = []
    for fp in track(file_paths[:100] + file_paths[-100:], 'features extraction'):
        _, file_name = path.split(fp)
        cls = file_name.split('.')[0]
        idx = map_class2index[cls]
        image = read_image(fp, size=(512, 512))
        tensor = cv2th(image)
        
        input_batch = tensor[None, ...]
        fingerprint = vectorizer(input_batch).squeeze(0)
        features_accumulator.append((th.flatten(fingerprint), idx))
    
    serialize(features_accumulator, 'features.pkl')
    logger.success('features were saved')

@router_command.command()
@click.option('--path2features')
@click.option('--nb_epochs', type=int)
@click.option('--bt_size', type=int)
def learning(path2features, nb_epochs, bt_size):
    features = deserialize(path2features)
    
    X = [ fingerprint for fingerprint, _ in features ]
    Y = [ label for _, label in features ]

    X = th.as_tensor(np.vstack(X) ).float() # size = 200x512 
    Y = th.as_tensor(Y).long()

    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=bt_size)

    net = NeuralNet()
    solver = th.optim.Adam(net.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(nb_epochs):
        for x_batch, y_batch in dataloader:
            output = net(x_batch)
            
            solver.zero_grad()
            error = criterion(output, y_batch)
            error.backward()
            solver.step()

            logger.debug(f'[{epoch:03d}:{nb_epochs:03d}]Loss : {error:07.3f}')
        # 
    # end loop for training ...! 

    th.save(net, 'predictor.th')

@router_command.command()
@click.option('--path2vectorizer')
@click.option('--path2image')
@click.option('--path2net')
def predict(path2vectorizer, path2image, path2net):
    labels = ['cat', 'dog']
    net = th.load(path2net)
    net.eval()

    vectorizer = load_vectorizer(path2vectorizer)   
    image = read_image(path2image, size=(512, 512))
    tensor = cv2th(image)
        
    input_batch = tensor[None, ...]
    fingerprint = th.flatten(vectorizer(input_batch))

    with th.no_grad():
        probabilities = net(fingerprint[None, ...])
        probabilities = th.softmax(probabilities, dim=1).squeeze(0)
        index = th.argmax(probabilities)
        logger.debug(f"it's a {labels[index]}")
        
        cv2.imshow('000', image)
        cv2.waitKey(0)

@router_command.command()
@click.option('--server_url')
@click.option('--path2image', type=click.Path(True))
def send_image(server_url, path2image):
    binary_stream = open(path2image, mode='rb')
    response = requests.post(server_url, files={'data': binary_stream})
    content = response.content
    print(content)

    
if __name__ == '__main__':
    router_command(obj={})