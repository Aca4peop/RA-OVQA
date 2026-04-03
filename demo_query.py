from pathlib import Path
from argparse import ArgumentParser
import json

import numpy as np
from PIL import Image
import faiss
import torch

from extractor import HierarchicalFeatureExtractor

MAIN_DIR = Path(__file__).resolve().parent


class VectorBaseQuery:
    def __init__(self, model, device, index_type):
        vecbase_dir = MAIN_DIR / "VectorBases"
        self.extractor = HierarchicalFeatureExtractor(model, device)
        self.index_type = index_type
        self.index = faiss.read_index(str(vecbase_dir / f"{model}_{index_type}.index"))
        self.feats_all = np.load(vecbase_dir / f"{model}_FeatsAll.npy")
        
        with open(vecbase_dir / "ImgNames.json", "r", encoding="utf-8") as f:
            self.imgnames = json.load(f)
    
    @torch.inference_mode()
    def querybyimg(self, pilimg:Image.Image, top_k:int=5):
        feat_all, feat_last = self.extractor.extract(pilimg)
        results = self.querybyfeat(feat_last, top_k)
        
        return feat_all.flatten(), results
    
    @torch.inference_mode()
    def querybyfeat(self, feat_last:np.ndarray, top_k:int=5):
        if self.index_type == "IP":
            faiss.normalize_L2(feat_last)
        
        scores, indices = self.index.search(feat_last, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "imgname": self.imgnames[idx],
                "sim_score": score,
                "feat_all": self.feats_all[idx]
            })
        
        return results

