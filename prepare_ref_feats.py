import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from tqdm import tqdm
from demo_query import VectorBaseQuery


if __name__ == "__main__":
    parser = ArgumentParser(description="Retrival similar viewports")
    parser.add_argument("--database", type=str, help="Database name.")
    parser.add_argument(
        "--frame_num",
        type=int,
        default=20,
        help="the number of required frames per video",
    )
    args = parser.parse_args()
    k = 5
    v_len = args.frame_num
    query = VectorBaseQuery("ConvNeXt_Base", "cuda", "IP")
    root_path = Path("./cache/features/" + args.database)
    srcs = os.listdir(root_path)

    save_path = Path("./cache/ref_features/" + args.database)
    save_path.mkdir(parents=True, exist_ok=True)
    for src in tqdm(srcs):
        feats = np.load(root_path / src)
        refs = np.zeros((v_len, 5, 1920))
        for i in range(0, v_len):
            feat = query.querybyfeat(feats[i : i + 1, -1024:], k)
            for j in range(k):
                refs[i, j] = feat[j]["feat_all"]
        np.save(save_path / src, refs)
