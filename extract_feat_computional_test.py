import os
from pathlib import Path
from argparse import ArgumentParser

from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
import torch
from extractor import HierarchicalFeatureExtractor


def extract_feat_video(extractor:HierarchicalFeatureExtractor, vidpath:Path, frame_num:int, max_idx:int, batch_size:int=10):
    vidcap = cv2.VideoCapture(str(vidpath))
    if not vidcap.isOpened():
        raise RuntimeError(f"Can't open {vidpath.name}!")
    vid_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_length = min(max_idx, vid_length)
    
    frames = []
    frames_idx = np.linspace(0, vid_length-1, frame_num, dtype=int)

    frame_idx = 0
    while True:
        ret, frame = vidcap.read()
        if not ret:
            break
        if frame_idx in frames_idx:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
            if frame_idx == frames_idx[-1]:
                break
        frame_idx += 1
    
    vidcap.release()

    frame_num = len(frames)

    feats_all = []
    feats_last = []
    for start_idx in range(0, frame_num, batch_size):
        end_idx = min(frame_num, start_idx+batch_size)
        frames_t = frames[start_idx:end_idx]
        feats_all_t, feats_last_t = extractor.extract(frames_t)  # [B, C]
        feats_all.append(feats_all_t)
        feats_last.append(feats_last_t)
    feats_all = np.concatenate(feats_all, axis=0)
    feats_last = np.concatenate(feats_last, axis=0)

    return feats_all, feats_last


def extract_feat_img(extractor: HierarchicalFeatureExtractor, VPSpath:Path,vname,VP_idx:int,
                       batch_size: int = 20):

    frames = []
    for i in range(20):
        img_path ='%s/%s_VP%d_F%d.png'%(VPSpath,vname[:-4],VP_idx,i)
        frames.append(Image.open(img_path))
    frame_num = len(frames)

    feats_all = []
    feats_last = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for start_idx in range(0, frame_num, batch_size):
        end_idx = min(frame_num, start_idx + batch_size)
        frames_t = frames[start_idx:end_idx]
        feats_all_t, feats_last_t = extractor.extract(frames_t)  # [B, C]
        start.record()
        feats_all_t, feats_last_t = extractor.extract(frames_t)  # [B, C]
        feats_all_t, feats_last_t = extractor.extract(frames_t)  # [B, C]
        feats_all_t, feats_last_t = extractor.extract(frames_t)  # [B, C]
        feats_all_t, feats_last_t = extractor.extract(frames_t)  # [B, C]
        feats_all_t, feats_last_t = extractor.extract(frames_t)  # [B, C]
        end.record()
        torch.cuda.synchronize()
        print(f"{vname} takes {start.elapsed_time(end)/5}ms")
        exit(0)
        feats_all.append(feats_all_t)
        feats_last.append(feats_last_t)
    feats_all = np.concatenate(feats_all, axis=0)
    feats_last = np.concatenate(feats_last, axis=0)

    return feats_all, feats_last


if __name__ == "__main__":
    parser = ArgumentParser(description="Extract Viewport Video Features")
    parser.add_argument("--model", type=str, default="ConvNeXt_Base", help="Backbone for feature extract")
    parser.add_argument("--database", type=str, default="VQA-ODV", help="one of [VQA-ODV, VRVQW]")
    parser.add_argument("--frame_num", type=int, default=20, help="the number of required frames per video")
    parser.add_argument("--max_idx", type=int, default=300, help="the maximum frame idx for video")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = HierarchicalFeatureExtractor(args.model, device)
    
    save_dir = Path(f"./cache/features/ODVmprnet")
    save_dir.mkdir(parents=True, exist_ok=True)

    # vidpaths= Path(f"./Viewports/{args.database}").glob("*.mp4")
    videos=os.listdir('/home2/ODV-half')
    VPSpath='/home1/hzy/PycharmProjects/ResShift/results/'

    for vname in tqdm(list(videos)):
        if not vname.endswith('.mp4'):
            continue
        if not ('ERP' in vname or 'RCMP' in vname or 'TSP' in vname):
            continue
        for i in range(5):
            feats_all, feats_last = extract_feat_img(extractor, VPSpath, vname, i,batch_size=10)
            # np.save('%s/%s_VP%d.npy'%(save_dir,vname[:-4],i), feats_all)