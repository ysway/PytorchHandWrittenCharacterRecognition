import torch.nn as nn
import torch.nn.functional as F
import torch

class CustomVGG19bn(nn.Module):
    def __init__(self, num_classes):
        super(CustomVGG19bn, self).__init__()
        self.features = nn.Sequential(*list(models.vgg19_bn(pretrained=True).features.children()))
        self.classifier = nn.Sequential(
            *[list(models.vgg19_bn(pretrained=True).classifier.children())[i] for i in range(6)],
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 512*7*7)
        out = self.classifier(x)
        return out