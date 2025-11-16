# 文件名: sort_tracker.py
# 作用: 实现一个简单的在线实时跟踪器 (SORT) (已修复 ValueError 和 IndexError)

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

def iou(bbox1, bbox2):
    """
    计算两个边界框的交并比 (IoU)。
    bbox: [x1, y1, x2, y2, ...] 
    """
    bb1_x1, bb1_y1, bb1_x2, bb1_y2 = bbox1[:4]
    bb2_x1, bb2_y1, bb2_x2, bb2_y2 = bbox2[:4]

    x_left = max(bb1_x1, bb2_x1)
    y_top = max(bb1_y1, bb2_y1)
    x_right = min(bb1_x2, bb2_x2)
    y_bottom = min(bb1_y2, bb2_y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1_x2 - bb1_x1) * (bb1_y2 - bb1_y1)
    bb2_area = (bb2_x2 - bb2_x1) * (bb2_y2 - bb2_y1)
    
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou

class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def convert_bbox_to_z(self, bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h
        r = w / float(h) if h != 0 else 0
        return np.array([x, y, s, r]).reshape((4, 1))

    def convert_x_to_bbox(self, x, score=None):
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w if w != 0 else 0
        if score is None:
            return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
        else:
            return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))

    def predict(self):
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return self.convert_x_to_bbox(self.kf.x)

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if len(trackers) == 0:
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk[:4])

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_indices = np.array(list(zip(row_ind, col_ind)))

    # --- 【修改点】: 增加对 matched_indices 是否为空的判断 ---
    if matched_indices.shape[0] > 0:
        # 筛选出 IoU 低于阈值的匹配
        matches_above_thresh = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] >= iou_threshold:
                matches_above_thresh.append(m.reshape(1,2))
        
        if len(matches_above_thresh) > 0:
            matches = np.concatenate(matches_above_thresh, axis=0)
            matched_det_indices = matches[:, 0]
            matched_trk_indices = matches[:, 1]
        else:
            matches = np.empty((0, 2), dtype=int)
            matched_det_indices = np.array([], dtype=int)
            matched_trk_indices = np.array([], dtype=int)
    else: # 如果没有任何匹配
        matches = np.empty((0, 2), dtype=int)
        matched_det_indices = np.array([], dtype=int)
        matched_trk_indices = np.array([], dtype=int)
    
    # 未匹配的检测就是所有检测中没有出现在匹配成功列表里的
    unmatched_detections = np.array([d for d in range(len(detections)) if d not in matched_det_indices])
    # 未匹配的跟踪器也是同理
    unmatched_trackers = np.array([t for t in range(len(trackers)) if t not in matched_trk_indices])
    
    # 筛选出 IoU 低于阈值但被匈牙利算法匹配上的那些，也归为未匹配
    if matched_indices.shape[0] > 0:
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                # 在某些实现中，这里可能需要将它们加入未匹配列表
                # 但更简洁的逻辑是，我们在开始就只考虑高于阈值的匹配
                pass

    return matches, unmatched_detections, unmatched_trackers

class Sort:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            self.trackers.append(trk)
            
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1))
            i -= 1
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        
        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0,5))