import torch
from torch import nn
from torchvision import models

class FlowerClassifier(nn.Module):
    def __init__(self, input_size, hidden_units, output_size, dropout_rate=0.5):
        super(FlowerClassifier, self).__init__()
        self.model = models.vgg13(pretrained=True)
        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Modify the classifier section of the VGG model
        self.model.classifier = nn.Sequential(
            nn.Linear(25088, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units, output_size),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = FlowerClassifier(
        input_size=checkpoint['input_size'],
        hidden_units=checkpoint['hidden_units'],
        output_size=checkpoint['output_size']
    )
    model.load_state_dict(checkpoint['state_dict'])
    
    return model
