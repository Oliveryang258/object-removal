# 文件名: process_video_with_detections.py
# (已修复 NameError)

import json
import argparse
import os
# --- 【核心修改点】: 导入 imageio 库 ---
import imageio
# ------------------------------------
from automated_video_remover import AutomatedVideoRemover

def main(args):
    print("--- Starting Automated Video Object Removal Pipeline ---")
    
    # 1. 加载检测结果
    print(f"Loading detections from: {args.detections_json_path}")
    with open(args.detections_json_path, 'r') as f:
        detections = json.load(f)

    # 2. 筛选出动态对象并按起始帧排序
    dynamic_objects = [d for d in detections if d.get('moving') == 1]
    dynamic_objects.sort(key=lambda x: x['start_frame_index'])

    if not dynamic_objects:
        print("No dynamic objects (moving=1) found in the detection file. Exiting.")
        return

    print(f"Found {len(dynamic_objects)} dynamic object(s) to remove.")

    # 3. 加载视频到内存
    print(f"Loading video from: {args.input_video_path}")
    try:
        # 使用 imageio.mimread 读取所有帧
        current_frames = imageio.mimread(args.input_video_path, memtest=False)
        print(f"Video loaded successfully with {len(current_frames)} frames.")
    except Exception as e:
        print(f"Error loading video: {e}")
        return
        
    # 使用 imageio.v3.immeta 读取元数据（如 fps）
    # 在较新版本的 imageio 中，推荐使用 v3 接口
    try:
        fps = imageio.v3.immeta(args.input_video_path, exclude_applied=False).get("fps", 30)
    except:
        # 备用方案，如果 v3 接口失败
        reader = imageio.get_reader(args.input_video_path)
        fps = reader.get_meta_data().get('fps', 30)
        reader.close()


    # 4. 初始化核心处理器
    model_paths = {
        'sam': args.sam_ckpt,
        'ostrack': args.tracker_ckpt,
        'sttn': args.vi_ckpt
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
    print(f"\nAll objects processed. Saving final video to: {args.output_video_path}")
    output_dir = os.path.dirname(args.output_video_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    imageio.mimwrite(args.output_video_path, current_frames, fps=fps, quality=8)
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