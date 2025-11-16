# 文件名: automated_video_remover.py

import torch
import torch.nn as nn
import numpy as np
import imageio.v2 as iio
from PIL import Image
import tempfile
import os
import cv2
from tqdm import tqdm
from typing import List
import sys
from pathlib import Path

# 动态添加子模块路径到系统路径，确保可以导入
sys.path.append(str(Path(__file__).resolve().parent / "pytracking"))

from sam_segment import build_sam_model
from ostrack import build_ostrack_model, get_box_using_ostrack
from sttn_video_inpaint import build_sttn_model, inpaint_video_with_builded_sttn
from pytracking.lib.test.evaluation.data import Sequence
from utils import dilate_mask

class AutomatedVideoRemover(nn.Module):
    def __init__(self, model_paths: dict, sam_model_type: str = "vit_h"):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing models on device: {self.device}")

        self.sam_predictor = build_sam_model(model_type=sam_model_type, ckpt_p=model_paths['sam'], device=self.device)
        self.ostrack_tracker = build_ostrack_model(tracker_param=model_paths['ostrack'])
        self.sttn_inpainter = build_sttn_model(model_type="sttn", ckpt_p=model_paths['sttn'], device=self.device)
        self.sttn_inpainter.to(self.device)
        print("All models loaded successfully.")

    @torch.no_grad()
    def remove_object(
        self,
        input_frames: List[np.ndarray],
        start_frame_index: int,
        initial_bbox: List[int],
        dilate_kernel_size: int = 15
    ) -> List[np.ndarray]:
        
        print(f"--- Generating initial mask on frame {start_frame_index}... ---")
        start_frame = input_frames[start_frame_index]
        sam_box = np.array([initial_bbox[0], initial_bbox[1], initial_bbox[0] + initial_bbox[2], initial_bbox[1] + initial_bbox[3]])
        
        self.sam_predictor.set_image(start_frame)
        masks, scores, _ = self.sam_predictor.predict(box=sam_box, multimask_output=True)
        self.sam_predictor.reset_image()
        
        initial_mask = masks[np.argmax(scores)]
        if dilate_kernel_size > 0:
            initial_mask = dilate_mask(initial_mask, dilate_kernel_size)
            
        tracker_bbox = cv2.boundingRect(initial_mask.astype(np.uint8))

        print("--> Step 1/3: Tracking the object...")
        frames_to_track = input_frames[start_frame_index:]
        
        with tempfile.TemporaryDirectory() as temp_dir_tracker:
            frame_paths_for_tracker = []
            for i, frame in enumerate(frames_to_track):
                path = os.path.join(temp_dir_tracker, f"{i:06d}.jpg")
                iio.imwrite(path, frame)
                frame_paths_for_tracker.append(path)

            seq = Sequence("temp_seq", frame_paths_for_tracker, 'inpaint-anything', np.array(tracker_bbox).reshape(1, 4))
            all_boxes = get_box_using_ostrack(self.ostrack_tracker, seq)

        print("--> Step 2/3: Generating masks for each subsequent frame...")
        all_masks = [initial_mask]
        ref_mask = initial_mask
        
        for i, frame in enumerate(tqdm(frames_to_track[1:], "Segmenting frames")):
            box = all_boxes[i + 1]
            sam_box = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]])

            self.sam_predictor.set_image(frame)
            masks, scores, _ = self.sam_predictor.predict(box=sam_box, multimask_output=True)
            self.sam_predictor.reset_image()

            mse = np.mean((masks.astype(np.int32) - ref_mask.astype(np.int32))**2, axis=(-2, -1))
            best_idx = mse.argmin()
            mask = masks[best_idx]
            
            if dilate_kernel_size > 0:
                mask = dilate_mask(mask, dilate_kernel_size)
            
            all_masks.append(mask)
            ref_mask = mask

        print("--> Step 3/3: Inpainting the video frames...")
        
        pil_frames = [Image.fromarray(f) for f in frames_to_track]
        pil_masks = [Image.fromarray(m.astype(np.uint8) * 255) for m in all_masks]

        inpainted_frames_pil = inpaint_video_with_builded_sttn(
            self.sttn_inpainter, pil_frames, pil_masks, device=self.device
        )
        
        inpainted_frames_np = [np.array(f) for f in inpainted_frames_pil]
        
        final_frames = input_frames[:start_frame_index] + inpainted_frames_np
        
        return final_frames