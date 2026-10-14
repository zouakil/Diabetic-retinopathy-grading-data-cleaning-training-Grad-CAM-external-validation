"""
gradcam.py - Grad-CAM heatmaps for the DR classifier.

PURPOSE
    Show which regions of a retinal image most influenced the model's score for
    a given class, as a sanity check that it looks at the retina (and ideally at
    lesions) and not at artifacts such as black corners or the camera notch.

METHOD (Selvaraju et al., 2017)
    1. forward pass, keep the activations A of the target layer (N channels)
    2. backward pass from the score of the chosen class, keep dScore/dA
    3. weight of channel k = average of its gradient over the spatial positions
    4. heatmap = ReLU( sum over k of weight_k * A_k ), upsampled to the image size
    5. normalized to [0, 1]

USAGE
    gradcam = GradCAM(model, model.bn2)           # bn2 = last feature map of EfficientNet-B0
    cam, probs, class_idx = gradcam(x)            # x: (1, 3, H, W) float32 tensor on the model's device
    cam is an (H, W) array in [0, 1]; class_idx defaults to the predicted class.

NOTES
    - The model must be in eval mode. Do NOT call under torch.no_grad() (gradients
      are needed) and do not use autocast.
    - The EfficientNet-B0 feature map is 16x16 for a 512x512 input, so heatmaps are
      coarse and show regions, not individual lesions.
    - A heatmap shows where the model looked, not why. It supports or contradicts
      a hypothesis, but it does not prove that the model "understands" lesions.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self._acts = None
        self._grads = None
        target_layer.register_forward_hook(self._save)

    def _save(self, module, inputs, output):
        """Keep the layer's output, and its gradient when the backward pass arrives."""
        self._acts = output.detach()
        output.register_hook(lambda g: setattr(self, "_grads", g.detach()))

    def __call__(self, x, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        if class_idx is None:
            class_idx = int(probs.argmax())
        logits[0, class_idx].backward()

        weights = self._grads.mean(dim=(2, 3), keepdim=True)         # one weight per channel
        cam = F.relu((weights * self._acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), probs.detach().cpu().numpy(), class_idx


def overlay(img, cam, alpha=0.4):
    """Blend a heatmap on an image. img: float RGB in [0, 1], (H, W, 3); cam: (H, W) in [0, 1]."""
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1] / 255.0
    return np.clip((1 - alpha) * img + alpha * heat, 0, 1)