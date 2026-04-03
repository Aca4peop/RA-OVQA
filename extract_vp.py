import json
import math
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
import skimage
import skvideo
from omni_utils import eq_to_pers


def extract_vp_video(data: Dict[str, Any], vid_root: Path, save_dir: Path,fov: float = 90.0, vp_size: Tuple[int, int] = (384, 384)):
    fov = np.deg2rad(fov)
    vidpath = vid_root / data["erpname"]
    viewports = data["viewports"]
    cods = [-3/4 * math.pi, -1/4 * math.pi, 1/4* math.pi, 3/4 * math.pi]
    if len(viewports) < 5:
        for i in range(5 - len(viewports)):
            viewports.append({"center": {"lon": cods[i], "lat": 0.0}})

    ffmpeg = skvideo.io.FFmpegReader(str(vidpath))
    print(f"Processing {vidpath.name}")

    frame_idx = 0
    save_idx = 0
    for frame in ffmpeg.nextFrame():
        if not frame_idx % 15==0:
            frame_idx += 1
            continue
        for idx, viewport in enumerate(viewports):
            lon, lat = viewport["center"].values()
            vp_img = eq_to_pers(frame, fov, lon, -lat, *vp_size)
            skimage.io.imsave('%s/%s_VP%d_F%d.png' % (save_dir, vidpath.stem, idx, save_idx), vp_img)

        frame_idx += 1
        save_idx += 1

    ffmpeg.close()


def extract_vp_frame(data:Dict[str, Any], vid_root:Path, save_dir:Path, fov:float=90.0, vp_size:Tuple[int, int]=(384, 384)):
    fov = np.deg2rad(fov)
    vidpath = vid_root / data["erpname"]
    viewports = data["viewports"]
    
    vid_cap = cv2.VideoCapture(str(vidpath))
    vid_fps = vid_cap.get(cv2.CAP_PROP_FPS)
    vid_length = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"Processing {vidpath.name}, {vid_length}x{vid_width}x{vid_height}@{vid_fps}fps")
    
    if not vid_cap.isOpened():
        raise RuntimeError(f"Can't open {vidpath.name}!")
    
    ret, frame = vid_cap.read()
    vid_cap.release()
    for idx, viewport in enumerate(viewports):
        lon, lat = viewport["center"].values()
        vp_img = eq_to_pers(frame, fov, lon, -lat, *vp_size)
        cv2.imwrite(str(save_dir / f"{vidpath.stem}_VP{idx+1:03d}.png"), vp_img)
    

def extract_vp_image(data:Dict[str, Any], img_root:Path, save_dir:Path,
                     fov:float=90.0, vp_size:Tuple[int, int]=(384, 384)):
    fov = np.deg2rad(fov)
    imgpath = img_root / data["erpname"]
    viewports = data["viewports"]

    erp_img = cv2.imread(str(imgpath))
    print(f"Processing {imgpath.name}, {erp_img.shape}")
    for idx, viewport in enumerate(viewports):
        lon, lat = viewport["center"].values()
        vp_img = eq_to_pers(erp_img, fov, lon, -lat, *vp_size)
        cv2.imwrite(str(save_dir / f"{imgpath.stem}_VP{idx+1:03d}.png"), vp_img)


if __name__ == "__main__":
    parser = ArgumentParser(description="Extract viewports according to centers")
    parser.add_argument("-i",type=str,help="Input path to the input ERP directory.")
    parser.add_argument("-o", type=str, help="Save path of viewports")
    parser.add_argument("-c", type=str, help="JSON file of viewport centers")
    parser.add_argument("--max_workers", type=int, default=16, help="max_workers for ProcessPoolExecutor()")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--video", action="store_true", help="Process video.")
    group.add_argument("--image", action="store_true", help="Process image.")
    args = parser.parse_args()
    assert args.c.endswith('.json')
    erppath = Path(args.i)
    save_dir = Path(args.o)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.c, "r", encoding="utf-8") as f:
        sam_results = json.load(f)["sam_results"]

    if args.image:
        extractor = partial(extract_vp_image, img_root=erppath, save_dir=save_dir)
    else:
        extractor = partial(extract_vp_frame, vid_root=erppath, save_dir=save_dir)


    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        executor.map(extractor, sam_results)

