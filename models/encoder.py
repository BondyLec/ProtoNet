import torch
import torch.nn as nn

class OmniglotEncoder(nn.Module):
    def __init__(self):
       super().__init__()

       self.encoder = nn.Sequential(
           nn.Conv2d(
               1, 64,
               kernel_size=3,
               padding=1
           ),
           nn.BatchNorm2d(64),
           nn.ReLU(),
           nn.MaxPool2d(2),

           nn.Conv2d(
               64, 64,
               kernel_size=3,
               padding=1
           ),
           nn.BatchNorm2d(64),
           nn.ReLU(),
           nn.MaxPool2d(2),

           nn.Conv2d(
               64, 64,
               kernel_size=3,
               padding=1
           ),
           nn.BatchNorm2d(64),
           nn.ReLU(),
           nn.MaxPool2d(2),

           nn.Conv2d(
               64, 64,
               kernel_size=3,
               padding=1
           ),
           nn.BatchNorm2d(64),
           nn.ReLU(),
           nn.MaxPool2d(2),
       )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        return x

