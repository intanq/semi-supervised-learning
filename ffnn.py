from torch import nn
import torch
    
class FFNN(nn.Module):
    def __init__(self, d_in=28*28, d_out=10):
        super(FFNN, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(d_in, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, d_out)
        )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits

class FFNN_v2(nn.Module):
    def __init__(self, d_in=28*28, d_out=10):
        super(FFNN_v2, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(d_in, 512),
            nn.ReLU(),
            nn.Linear(512, 300),
            nn.ReLU(),
            nn.Linear(300, d_out)
        )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits