# 文件名: dynamic_object_detector.py
# 作用: 核心脚本，负责检测、跟踪、判断动态并输出JSON

import cv2
from ultralytics import YOLO
import numpy as np
import math
import json
import argparse
from sort_tracker import Sort  # 导入我们刚刚创建的跟踪器

def main(args):
    # ---------------- 初始化 YOLO 模型和 SORT 跟踪器 ----------------
    print(f"Loading YOLO model from: {args.model_path}")
    model = YOLO(args.model_path)
    tracker = Sort(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.iou_threshold)

    # ---------------- 视频输入初始化 ----------------
    # 摄像头ID可以是数字，文件路径是字符串
    video_input = int(args.video_path) if args.video_path.isdigit() else args.video_path
    cap = cv2.VideoCapture(video_input)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {args.video_path}")
        return

    # ---------------- 状态变量初始化 ----------------
    frame_index = -1
    tracked_objects_history = {}  # key: track_id, value: {'last_center': (x,y), 'is_moving': bool, 'frames_since_seen': int}
    dynamic_objects_info = {}     # key: track_id, value: a dict with object info for JSON output

    print("Processing video... Press 'q' to stop.")

    # ---------------- 主循环：处理视频流 ----------------
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame.")
            break
        
        frame_index += 1

        # 1. YOLOv8 推理
        results = model(frame, conf=args.conf_threshold, verbose=False)
        
        # 2. 格式化检测结果以适配 SORT
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            score = box.conf[0].item()
            detections.append([x1, y1, x2, y2, score])
        
        detections_np = np.array(detections) if detections else np.empty((0, 5))

        # 3. 更新 SORT 跟踪器
        tracked_result = tracker.update(detections_np) # 返回 [x1, y1, x2, y2, track_id]

        # 4. 判断动态对象
        current_frame_track_ids = set()
        if tracked_result.size > 0:
            for track in tracked_result:
                x1, y1, x2, y2, track_id = track
                track_id = int(track_id)
                current_frame_track_ids.add(track_id)
                
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                if track_id in tracked_objects_history:
                    # 这是一个已知的对象，计算位移
                    prev_center_x, prev_center_y = tracked_objects_history[track_id]['last_center']
                    displacement = math.sqrt((center_x - prev_center_x)**2 + (center_y - prev_center_y)**2)
                    
                    if displacement > args.movement_threshold:
                        tracked_objects_history[track_id]['is_moving'] = True
                    
                    # 如果对象被确认为动态，并且尚未记录，则记录其初始信息
                    if tracked_objects_history[track_id]['is_moving'] and track_id not in dynamic_objects_info:
                        print(f"Detected dynamic object with ID: {track_id} at frame {frame_index}")
                        dynamic_objects_info[track_id] = {
                            "object_id": track_id,
                            "moving": 1,
                            "start_frame_index": frame_index,
                            "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                        }
                    
                    # 更新历史信息
                    tracked_objects_history[track_id]['last_center'] = (center_x, center_y)
                    tracked_objects_history[track_id]['frames_since_seen'] = 0
                else:
                    # 这是一个新出现的对象
                    tracked_objects_history[track_id] = {
                        'last_center': (center_x, center_y),
                        'is_moving': False,
                        'frames_since_seen': 0
                    }

        # 5. 管理丢失的跟踪对象
        lost_ids = []
        for track_id in tracked_objects_history:
            if track_id not in current_frame_track_ids:
                tracked_objects_history[track_id]['frames_since_seen'] += 1
                if tracked_objects_history[track_id]['frames_since_seen'] > args.max_age:
                    lost_ids.append(track_id)
        
        for track_id in lost_ids:
            del tracked_objects_history[track_id]

        # (可选) 可视化
        if args.show:
            # 绘制 YOLO 框
            annotated_frame = results[0].plot()
            # 绘制跟踪器 ID
            if tracked_result.size > 0:
                for track in tracked_result:
                    x1, y1, x2, y2, track_id = map(int, track)
                    cv2.putText(annotated_frame, f"ID: {track_id}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Dynamic Object Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # ---------------- 循环结束，生成输出 ----------------
    cap.release()
    if args.show:
        cv2.destroyAllWindows()

    output_data = list(dynamic_objects_info.values())
    
    with open(args.output_json, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"\n✅ Dynamic object detection complete. Results saved to {args.output_json}")
    print(f"Found {len(output_data)} dynamic objects.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamic Object Detector using YOLO and SORT")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the video file or camera ID.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the YOLO .pt model file.")
    parser.add_argument("--output_json", type=str, default="detections.json", help="Path to save the output JSON file.")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="YOLO confidence threshold.")
    parser.add_argument("--movement_threshold", type=float, default=5.0, help="Pixel displacement threshold to consider an object as 'moving'.")
    
    # SORT 参数
    parser.add_argument("--max_age", type=int, default=30, help="SORT: Maximum number of frames to keep a track without a detection.")
    parser.add_argument("--min_hits", type=int, default=3, help="SORT: Minimum number of hits to start a track.")
    parser.add_argument("--iou_threshold", type=float, default=0.3, help="SORT: IoU threshold for association.")
    
    parser.add_argument("--show", action='store_true', help="Show the video with detections and tracking IDs.")
    
    args = parser.parse_args()
    main(args)