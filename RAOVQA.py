'''
This code is directly used for quality prediction of a video

parameters: video_path: The path of a video

'''
import shutil
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import skvideo.io
import os
import numpy as np
import skvideo.io

from extract_feat import extract_feat_client
from extract_vp import extract_vp_video_client
from get_center import tinysam_client
from models.QAmodel import GatedMoM
from prepare_ref_feats import query_client

warnings.filterwarnings("ignore")



def inference(video_path):
    os.mkdir('tmp')
    device = torch.device("cuda")
    tinysam_client(video_path)
    extract_vp_video_client(video_path)
    extract_feat_client(video_path)
    query_client()
    vname = Path(video_path).name
    feat = []
    refs = []
    for i in range(0, 5):
        if os.path.exists("./tmp/features/"+ vname[:-4] + "_VP" + str(i) + ".npy"):
            feat.append(
                np.load("./tmp/features/" + vname[:-4] + "_VP" + str(i) + ".npy")[np.newaxis, :])
            refs.append(
                np.load("./tmp/ref_features/" + vname[:-4] + "_VP" + str(i) + ".npy")[np.newaxis,
                :])
    feat = np.concatenate(feat, axis=0)
    refs = np.concatenate(refs, axis=0)
    feat=torch.from_numpy(feat).float().to(device)
    refs = torch.from_numpy(refs).float().to(device)


    model = GatedMoM().to(device)
    model.load_state_dict(torch.load('./model.pth'))
    model.eval()
    pred, _, _, _ = model(feat, refs)
    pred = pred.mean().view(-1).to("cpu").numpy()
    shutil.rmtree("./tmp")
    return 1 - pred


if __name__ == "__main__":
    video_file = sys.argv[1]
    score = inference(video_file)
    print('quality prediction: %.6f' % score)


