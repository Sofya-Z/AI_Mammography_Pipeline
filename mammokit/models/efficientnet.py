"""EfficientNet classifier with a replaceable head."""

from operator import attrgetter

from torch import nn
from torchvision import models


class EfficientNetClassifier(nn.Module):
    def __init__(self, backbone="efficientnet_b3", weights="EfficientNet_B3_Weights.DEFAULT", num_classes=2):
        super().__init__()

        if weights is None:
            self.net = getattr(models, backbone)(weights=None)
        else:
            self.net = getattr(models, backbone)(weights=attrgetter(weights)(models))

        if num_classes is not None:
            in_features = self.net.classifier[1].in_features
            self.net.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.net(x)
