import torch as th
import torch.nn as nn 
import torch.nn.functional as F 

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()  # 
        self.layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    
    def forward(self, X):
        return self.layers(X)