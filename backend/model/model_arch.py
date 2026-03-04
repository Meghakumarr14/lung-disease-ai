import timm
import torch.nn as nn

def create_model(num_classes=5):
    """
    Creates SENet154 model architecture
    """
    model = timm.create_model(
        'senet154',
        pretrained=False,
        num_classes=num_classes
    )
    return model
