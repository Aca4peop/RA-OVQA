import json
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm
import faiss
import torch
from PIL import Image

from extractor import HierarchicalFeatureExtractor


if __name__ == "__main__":
    parser = ArgumentParser(description="Build Faiss Vector Databse on CustomERPframes")
    parser.add_argument("--model", type=str, default="ConvNeXt_Base", help="Backbone for feature extract")
    parser.add_argument("--dir", type=str, help="Path of viewports")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    imgpaths = Path(args.dir).glob("*.png")
    extractor = HierarchicalFeatureExtractor(args.model, device)
    
    imgnames, feats_all, feats_last = [], [], []
    for imgpath in tqdm(list(imgpaths)):
        img = Image.open(imgpath)
        feat_all, feat_last = extractor.extract(img)  # [1, C]
        imgnames.append(imgpath.name)
        feats_all.append(feat_all)
        feats_last.append(feat_last)
    
    save_dir = Path("./VectorBases")
    save_dir.mkdir(parents=True, exist_ok=True)
    feats_all = np.concatenate(feats_all, axis=0)
    feats_last = np.concatenate(feats_last, axis=0)
    print(f"feats_all.shape: {feats_all.shape}")
    print(f"feats_last.shape: {feats_last.shape}")
    np.save(save_dir / f"{args.model}_FeatsAll.npy", feats_all)
    np.save(save_dir / f"{args.model}_FeatsLast.npy", feats_last)
    with open(save_dir / "ImgNames.json", "w", encoding="utf-8") as f:
        json.dump(imgnames, f, ensure_ascii=False, indent=4)
    
    index_l2 = faiss.IndexFlatL2(feats_last.shape[1])
    index_l2.add(feats_last)
    faiss.write_index(index_l2, str(save_dir / f"{args.model}_L2.index"))

    faiss.normalize_L2(feats_last)
    index_ip = faiss.IndexFlatIP(feats_last.shape[1])  # Inner Product
    index_ip.add(feats_last)
    faiss.write_index(index_ip, str(save_dir / f"{args.model}_IP.index"))
