"""
model.py - Model definition.

create_model builds a pretrained EfficientNet (timm) whose last layer is replaced
by a 5-class output, one per DR grade, with dropout 0.3.

NOTES
    - pretrained=True downloads the ImageNet weights (cached after the first
      download). When loading our own trained weights (.pth), use
      create_model(pretrained=False): the weights are overwritten anyway.
    - The model outputs raw scores (logits). Apply softmax to get probabilities.
"""
import timm


def create_model(name="efficientnet_b0", num_classes=5, pretrained=True):
    """Pretrained EfficientNet with a new 5-class output layer."""
    return timm.create_model(
        name,
        pretrained=pretrained,     # download weights learned on ImageNet
        num_classes=num_classes,   # replaces the 1000-class last layer by a 5-class one
        drop_rate=0.3,             # dropout: randomly switches off 30% of units to reduce overfitting
    )