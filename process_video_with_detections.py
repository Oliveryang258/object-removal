# 文件名: process_video_with_detections.py
# (已修复 FileNotFoundError)

import json
import argparse
import os
import imageio
from automated_video_remover import AutomatedVideoRemover
from pathlib import Path # 导入 Path 对象

def main(args):
    print("--- Starting Automated Video Object Removal Pipeline ---")
    
    # --- 【核心修改点】: 将所有相对路径转换为绝对路径 ---
    # 获取当前脚本所在的目录
    script_dir = Path(__file__).resolve().parent
    
    # 基于脚本目录构建绝对路径
    input_video_path = script_dir / args.input_video_path
    detections_json_path = script_dir / args.detections_json_path
    output_video_path = script_dir / args.output_video_path
    sam_ckpt_path = script_dir / args.sam_ckpt
    vi_ckpt_path = script_dir / args.vi_ckpt
    # tracker_ckpt 是一个名字，不是路径，所以不需要转换

    # 1. 加载检测结果
    print(f"Loading detections from: {detections_json_path}")
    with open(detections_json_path, 'r') as f:
        detections = json.load(f)

    # 2. 筛选出动态对象并按起始帧排序
    dynamic_objects = [d for d in detections if d.get('moving') == 1]
    dynamic_objects.sort(key=lambda x: x['start_frame_index'])

    if not dynamic_objects:
        print("No dynamic objects (moving=1) found in the detection file. Exiting.")
        return

    print(f"Found {len(dynamic_objects)} dynamic object(s) to remove.")

    # 3. 加载视频到内存
    print(f"Loading video from: {input_video_path}")
    try:
        current_frames = imageio.mimread(input_video_path, memtest=False)
        print(f"Video loaded successfully with {len(current_frames)} frames.")
    except Exception as e:
        print(f"Error loading video: {e}")
        return
        
    try:
        fps = imageio.v3.immeta(input_video_path, exclude_applied=False).get("fps", 30)
    except:
        reader = imageio.get_reader(input_video_path)
        fps = reader.get_meta_data().get('fps', 30)
        reader.close()

    # 4. 初始化核心处理器
    model_paths = {
        'sam': str(sam_ckpt_path), # 传入绝对路径字符串
        'ostrack': args.tracker_ckpt, # tracker_ckpt 保持不变
        'sttn': str(vi_ckpt_path)   # 传入绝对路径字符串
    }
    remover = AutomatedVideoRemover(model_paths, sam_model_type=args.sam_model_type)

    # 5. 串行移除所有动态对象
    for i, obj in enumerate(dynamic_objects):
        print(f"\n--- Processing object {i+1}/{len(dynamic_objects)} (ID: {obj.get('object_id', 'N/A')}) ---")
        
        if obj['start_frame_index'] >= len(current_frames):
            print(f"Warning: start_frame_index {obj['start_frame_index']} is out of bounds for a video with {len(current_frames)} frames. Skipping object.")
            continue
            
        current_frames = remover.remove_object(
            input_frames=current_frames,
            start_frame_index=obj['start_frame_index'],
            initial_bbox=obj['bbox'],
            dilate_kernel_size=args.dilate_kernel_size
        )
        print(f"--- Finished processing object {i+1}/{len(dynamic_objects)} ---")

    # 6. 保存最终视频
    print(f"\nAll objects processed. Saving final video to: {output_video_path}")
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
        
    imageio.mimwrite(str(output_video_path), current_frames, fps=fps, quality=8)
    print("✅✅✅ Pipeline complete! ✅✅✅")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Automated Video Object Removal Pipeline")
    
    parser.add_argument("--input_video_path", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--detections_json_path", type=str, required=True, help="Path to the JSON file with object detections.")
    parser.add_argument("--output_video_path", type=str, required=True, help="Path to save the output video.")

    parser.add_argument("--sam_model_type", type=str, default="vit_h", choices=['vit_h', 'vit_l', 'vit_b', 'vit_t'], help="SAM model type.")
    parser.add_argument("--sam_ckpt", type=str, required=True, help="Path to the SAM checkpoint.")
    parser.add_argument("--tracker_ckpt", type=str, required=True, help="Parameter name of the tracker checkpoint.")
    parser.add_argument("--vi_ckpt", type=str, required=True, help="Path to the video inpainting (STTN) checkpoint.")
    
    parser.add_argument("--dilate_kernel_size", type=int, default=15, help="Kernel size for mask dilation.")

    args = parser.parse_args()
    main(args)