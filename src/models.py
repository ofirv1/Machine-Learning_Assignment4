from __future__ import annotations

import contextlib
import io
import os
import sys

import torch
import torch.nn as nn
from torchvision import models


def build_vgg19_model(num_classes: int) -> nn.Module:
    model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

    # freeze backbone
    for p in model.features.parameters():
        p.requires_grad = False

    # replace final classifier layer
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


class YOLOv5Backbone(nn.Module):
    
    #Loads YOLOv5 via torch.hub and extracts features via a forward hook.

    def __init__(self, yolo_variant: str = "yolov5s", pretrained: bool = True, freeze: bool = True):
        super().__init__()

        os.environ["YOLOv5_VERBOSE"] = "False"
        os.environ["ULTRALYTICS_VERBOSE"] = "False"

        # silence hub prints
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            hub_model = torch.hub.load("ultralytics/yolov5", yolo_variant, pretrained=pretrained)

            # in some envs, YOLO defines LOGGER in utils.general
            if "utils.general" in sys.modules and hasattr(sys.modules["utils.general"], "LOGGER"):
                sys.modules["utils.general"].LOGGER.setLevel(50)

        self.det = hub_model.model  # YOLO model object

        if freeze:
            for p in self.det.parameters():
                p.requires_grad = False

        self._feat = None

        def _hook(_, __, output):
            self._feat = output

        stack = self.det.model.model
    
        stack[-2].register_forward_hook(_hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._feat = None
        _ = self.det(x)
        if self._feat is None:
            raise RuntimeError("Backbone feature extraction failed (hook returned None).")
        return self._feat


class YOLOv5Classifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 102,
        yolo_variant: str = "yolov5s",
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = YOLOv5Backbone(
            yolo_variant=yolo_variant,
            pretrained=pretrained,
            freeze=freeze_backbone
        )
        self.classifier: nn.Linear | None = None
        self.num_classes = num_classes

    def init_head(self, img_size: int, device: torch.device) -> None:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size, device=device)
            feat = self.backbone(dummy)              # [1, C, H, W]
            feat_dim = feat.mean(dim=(2, 3)).shape[1]  # C

        self.classifier = nn.Linear(feat_dim, self.num_classes).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.classifier is None:
            raise RuntimeError("YOLOv5Classifier head not initialized. Call init_head(img_size, device).")
        feat = self.backbone(x)
        pooled = feat.mean(dim=(2, 3))
        logits = self.classifier(pooled)
        return logits


def unfreeze_last_yolo_layers(model_yolo: YOLOv5Classifier, n_last: int = 30) -> None:
   
    #Unfreeze last n_last layers from YOLO backbone stack.

    layers = list(model_yolo.backbone.det.model.model)
    for layer in layers[-n_last:]:
        for p in layer.parameters():
            p.requires_grad = True
