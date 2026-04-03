import json
from pathlib import Path
from argparse import ArgumentParser
from typing import Union, List, Tuple, Dict, Any

import cv2
import torch
import numpy as np
from tqdm import tqdm

import sys
sys.path.append("..")
from tinysam import sam_model_registry, SamHierarchicalMaskGenerator


def contains(bbox_large:List[float], bbox_small:List[float]) -> bool:
        x0_large, y0_large = min(bbox_large[0], bbox_large[2]), min(bbox_large[1], bbox_large[3])
        x1_large, y1_large = max(bbox_large[0], bbox_large[2]), max(bbox_large[1], bbox_large[3])
        
        x0_small, y0_small = min(bbox_small[0], bbox_small[2]), min(bbox_small[1], bbox_small[3])
        x1_small, y1_small = max(bbox_small[0], bbox_small[2]), max(bbox_small[1], bbox_small[3])

        return x0_small >= x0_large and x1_small <= x1_large \
            and y0_small >= y0_large and y1_small <= y1_large


def get_viewport(mask_generator: SamHierarchicalMaskGenerator, erppath:Union[str | Path],
                 max_vp:int=5, min_ratio:float=0.005, new_size:Tuple[int, int] | None=None) -> List[Dict[str, Any]]:
    """
    Extract `max_vp` viewports from an omnidirectional image/video according to the predition of tinysam
    
    :param mask_generator: mask generator initilized with pretrained tinysam model
    :param erppath: path to omni image/video, str or Path
    :param max_vp: max viewports to be extracted
    :param min_ratio: min area ratio of object
    :param new_size: (W, H), input resolution to tinysam
    
    :return list of segmentation results and centers
    """
    erppath = Path(erppath)
    if erppath.name.endswith(".mp4") or erppath.name.endswith(".mkv"):
        vidcap = cv2.VideoCapture(str(erppath))
        if not vidcap.isOpened():
            raise RuntimeError(f"Can't open {erppath.name}!")
        
        if erppath.name in ["p_JourneyOfSpace.mp4", "B_Cliff.mp4", "K_RollerCoaster1.mp4"]:  # 第一帧为全黑帧
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, 30)
        
        ret, img = vidcap.read()
        vidcap.release()
        if not ret:
            raise RuntimeError(f"Can't read the first frame of {erppath.name}!")
    
    elif erppath.name.endswith(".png") or erppath.name.endswith(".jpg"):
        img = cv2.imread(str(erppath))
    else:
        raise RuntimeError("Unsupported file format!")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if new_size is not None:
        img = cv2.resize(img, new_size)
    
    masks = mask_generator.hierarchical_generate(img)
    sorted_masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)
    
    selected_bboxes = []
    min_area = min_ratio * new_size[0] * new_size[1]

    for mask in sorted_masks:
        if len(selected_bboxes) >= max_vp or mask["area"] <= min_area:
            break

        if any(contains(bbox, mask["bbox"]) for bbox in selected_bboxes):
            continue
        
        selected_bboxes.append(mask["bbox"])

    result = []
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for bbox in selected_bboxes:
        x0, y0 = int(min(bbox[0], bbox[2])), int(min(bbox[1], bbox[3]))
        x1, y1 = int(max(bbox[0], bbox[2])), int(max(bbox[1], bbox[3]))
        # img = cv2.rectangle(img, (x0,y0), (x1, y1), color=(0, 255, 0), thickness=2)
        
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        # img = cv2.drawMarker(img, (int(x_center), int(y_center)), color=(0, 255, 0),
        #                      markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2, line_type=cv2.LINE_AA)
        
        lon = (2 * x_center - new_size[0]) / (new_size[0]) * np.pi
        lat = (0.5 - y_center / new_size[1]) * np.pi
        
        result.append({
            "box_position": {
                "upperleft": [x0, y0],
                "lowerright": [x1, y1]
            },
            "center": {
                "longitude": lon,
                "latitude": lat
            }
        })

    if len(result) == 0:
        # img = cv2.rectangle(img, (0, 0), (new_size[0]-1, new_size[1]-1), color=(0, 255, 0), thickness=2)
        # img = cv2.drawMarker(img, (new_size[0] // 2, new_size[1] // 2), color=(0, 255, 0),
        #                      markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2, line_type=cv2.LINE_AA)
        result.append({
            "box_position": {
                "upperleft": [0, 0],
                "lowerright": [new_size[0]-1, new_size[1]-1]
            },
            "center": {
                "longitude": 0,
                "latitude": 0
            }
        })
    
    # if savedir is not None:
    #     cv2.imwrite(str(savedir / f"{erppath.stem}.png"), img)

    return result


if __name__ == "__main__":
    parser = ArgumentParser(description="Extract viewport center from ERP images/videos using TinySAM")
    parser.add_argument("-i", type=str, help="Input path to the input ERP directory.")
    parser.add_argument("-o",  type=str, help="Save path to viewport centers, ends with .json")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--video", action="store_true", help="Process video.")
    group.add_argument("--image", action="store_true", help="Process image.")
    args = parser.parse_args()
    
    erppath = Path(args.i)
    if not args.o.endswith(".json"):
        args.o = args.o + "vp_center.json"

    if args.image:
        files = [f for f in erppath.rglob("*") if f.suffix.lower() in [".png", ".jpg"]]
    else:
        files = [f for f in erppath.rglob("*") if f.suffix.lower() in [".mp4", ".mkv"]]
    
    model_type = "vit_t"
    sam = sam_model_registry[model_type](checkpoint="./weights/tinysam_42.3.pth")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device=device)
    sam.eval()
    mask_generator = SamHierarchicalMaskGenerator(sam)
    new_size = (1024, 512)

    results = []
    savedir = Path(args.o).parent
    savedir.mkdir(parents=True, exist_ok=True)
    
    for erppath in tqdm(list(files)):
        erpname = erppath.name
        result = get_viewport(mask_generator, erppath, new_size=new_size)
        results.append({
            "erpname": erpname,
            "viewports": result
        })
    
    data = {
        "model": "TinySAM(Vit-T)",
        "resolution": new_size,
        "sam_results": results
    }

    with open(args.o, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)