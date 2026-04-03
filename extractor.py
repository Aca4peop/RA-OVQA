from pathlib import Path
from typing import List

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import timm


class HierarchicalFeatureExtractor:
    """
    Extract Hierarchical features from a pre-trained model.
    """
    def __init__(self, model, device):
        match model:
            case "ConvNeXt_Base":
                self.model = timm.create_model(
                    "convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384",
                    pretrained=True,
                    features_only=True,
                )
            case _:
                raise RuntimeError("Unsupported Model!")
        
        self.model = self.model.eval().to(device)
        # get model specific transforms (normalization, resize)
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

    @torch.inference_mode()
    def extract(self, pilimg:Image.Image | List[Image.Image]) -> np.ndarray:
        device = next(self.model.parameters()).device
        if isinstance(pilimg, Image.Image):
            img_t = self.transforms(pilimg).unsqueeze(0).to(device) # unsqueeze single image into batch of 1
            feats = self.model(img_t)
        elif isinstance(pilimg, list) and all(isinstance(img, Image.Image) for img in pilimg):
            img_t = [self.transforms(img) for img in pilimg]
            img_t = torch.stack(img_t, dim=0).to(device)
            feats = self.model(img_t)
        else:
            raise TypeError("pilimg must be PIL.Image.Image or List[PIL.Image.Image]")
        
        B = feats[0].shape[0]
        feat_all = torch.Tensor()
        for feat in feats:
            pooled_feat = self.pooling(feat).view(B, -1).detach().cpu()
            feat_all = torch.concatenate((feat_all, pooled_feat), dim=1)
        feat_all = feat_all.numpy()
        feat_last = pooled_feat.numpy()
        
        return feat_all, feat_last