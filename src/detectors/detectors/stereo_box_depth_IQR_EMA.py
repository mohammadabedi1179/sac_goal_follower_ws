#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String
import json
from collections import deque
import statistics
import numpy as np
import time

# === Camera Parameters (verify with your calibration!) ===
FOCAL_PX = 555.78        # From your working goal marker depth node
BASELINE_M = 0.0717      # ← REAL optical baseline = 7.17 cm

# === Tuning Parameters ===
MAX_TIME_DIFF = 0.05       # Very strict sync (was 0.1)
CONF_THRESH = 0.35
MIN_DISPARITY_PX = 3.0
DEPTH_MIN = 0.3
DEPTH_MAX = 50.0

# OptiRSDE-style parameters
BUFFER_LEN = 12            # Longer buffer = better IQR filtering
EMA_ALPHA = 0.3            # Smoothing factor (0.3 from paper)
EMA_UPDATE_EVERY = 10      # Apply EMA every 10 updates per object
OBJECT_EXPIRATION_S = 15.0 # Forget object if not seen for 15 seconds


class StereoBoxDepthNode(Node):
    def __init__(self):
        super().__init__('stereo_box_depth', namespace='follower_robot')

        self.left_sub = self.create_subscription(
            Detection2DArray,
            '/follower_robot/depth_cam/left/detections',
            self.left_cb,
            10
        )
        self.right_sub = self.create_subscription(
            Detection2DArray,
            '/follower_robot/depth_cam/right/detections',
            self.right_cb,
            10
        )

        self.depth_pub = self.create_publisher(
            String,
            '/follower_robot/obstacles_depth',
            10
        )

        self.last_left = None
        self.last_right = None

        # Per-object state: track_id (or class as fallback) → state dict
        self.objects = {}

        self.get_logger().info('Advanced stereo_box_depth with EMA + IQR + Tracking READY')

    def left_cb(self, msg: Detection2DArray):
        self.last_left = msg
        self.try_match()

    def right_cb(self, msg: Detection2DArray):
        self.last_right = msg
        self.try_match()

    def try_match(self):
        if not (self.last_left and self.last_right):
            return

        # Strict timestamp sync
        lt = self.last_left.header.stamp.sec + self.last_left.header.stamp.nanosec * 1e-9
        rt = self.last_right.header.stamp.sec + self.last_right.header.stamp.nanosec * 1e-9
        if abs(lt - rt) > MAX_TIME_DIFF:
            return

        results = []
        current_time = time.time()

        for ld in self.last_left.detections:
            lbox = self._bbox(ld)
            lcls = self._cls(ld)
            lconf = self._conf(ld)
            track_id = ld.id if ld.id else None

            if not (lbox and lcls) or (lconf and lconf < CONF_THRESH):
                continue

            # Find best right match using IoU + vertical alignment
            best_rd = self._find_best_match(ld, self.last_right.detections)
            if not best_rd:
                continue

            rbox = self._bbox(best_rd)

            # === Robust disparity from 4 corners + per-frame IQR ===
            disparity = self._robust_disparity_4corner_iqr(lbox, rbox)
            if not disparity or disparity < MIN_DISPARITY_PX:
                continue

            # === Per-object key (prefer track_id) ===
            obj_key = track_id if track_id else lcls

            # === Get or create object state ===
            if obj_key not in self.objects:
                self.objects[obj_key] = {
                    'smoothed_disp': None,
                    'buffer': deque(maxlen=BUFFER_LEN),
                    'update_count': 0,
                    'last_seen': current_time
                }

            state = self.objects[obj_key]
            state['last_seen'] = current_time

            # Expire old objects
            if current_time - state['last_seen'] > OBJECT_EXPIRATION_S:
                state['smoothed_disp'] = None
                state['buffer'].clear()
                state['update_count'] = 0

            # === IQR filtering on historical buffer ===
            state['buffer'].append(disparity)
            filtered_disp = self._iqr_filter(list(state['buffer']))

            current_disp = filtered_disp if filtered_disp is not None else disparity

            # === EMA update (every N frames per object) ===
            state['update_count'] += 1
            if state['update_count'] % EMA_UPDATE_EVERY == 0:
                if state['smoothed_disp'] is None:
                    state['smoothed_disp'] = current_disp
                else:
                    state['smoothed_disp'] = (
                        EMA_ALPHA * current_disp +
                        (1 - EMA_ALPHA) * state['smoothed_disp']
                    )

            final_disp = state['smoothed_disp'] if state['smoothed_disp'] is not None else current_disp

            if final_disp < MIN_DISPARITY_PX:
                continue

            depth = FOCAL_PX * BASELINE_M / final_disp
            if not (DEPTH_MIN <= depth <= DEPTH_MAX):
                continue

            results.append({
                'id': track_id,
                'class': lcls,
                'disparity_px': round(final_disp, 3),
                'depth_m': round(depth, 3),
                'raw_disp': round(disparity, 3)
            })

        # Cleanup expired objects
        expired = [k for k, v in self.objects.items() if current_time - v['last_seen'] > OBJECT_EXPIRATION_S]
        for k in expired:
            del self.objects[k]

        if results:
            msg = String()
            msg.data = json.dumps(results, separators=(',', ':'))
            self.depth_pub.publish(msg)
            self.get_logger().info(f'object depths: {msg.data}')

    def _find_best_match(self, left_det, right_dets):
        lbox = self._bbox(left_det)
        best_iou = 0.0
        best_det = None
        lcls = self._cls(left_det)

        for rd in right_dets:
            if self._conf(rd) and self._conf(rd) < CONF_THRESH:
                continue
            if lcls and self._cls(rd) and self._cls(rd) != lcls:
                continue

            rbox = self._bbox(rd)
            if not rbox:
                continue

            iou = self._bbox_iou(lbox, rbox)
            if iou > best_iou and iou > 0.3:
                best_iou = iou
                best_det = rd

        return best_det

    @staticmethod
    def _robust_disparity_4corner_iqr(lbox, rbox):
        lx1, ly1, lx2, ly2 = lbox
        rx1, ry1, rx2, ry2 = rbox

        corners_l = [(lx1, ly1), (lx2, ly1), (lx1, ly2), (lx2, ly2)]
        corners_r = [(rx1, ry1), (rx2, ry1), (rx1, ry2), (rx2, ry2)]

        disparities = []
        for (xl, yl), (xr, yr) in zip(corners_l, corners_r):
            if abs(yl - yr) < 8.0:  # Epipolar tolerance
                disparities.append(abs(xl - xr))

        if len(disparities) < 2:
            return None

        # Per-frame IQR (like OptiRSDE keypoints)
        sorted_d = sorted(disparities)
        q1, q3 = np.percentile(sorted_d, [25, 75])
        iqr_val = q3 - q1
        lower = q1 - 1.5 * iqr_val
        upper = q3 + 1.5 * iqr_val
        filtered = [d for d in disparities if lower <= d <= upper]

        return statistics.mean(filtered) if filtered else statistics.mean(disparities)

    @staticmethod
    def _iqr_filter(values):
        if len(values) < 3:
            return None
        arr = np.array(values)
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        filtered = arr[(arr >= lower) & (arr <= upper)]
        return float(filtered.mean()) if len(filtered) > 0 else None

    @staticmethod
    def _bbox_iou(box1, box2):
        x1, y1, x2, y2 = box1
        x1r, y1r, x2r, y2r = box2
        inter_x1 = max(x1, x1r)
        inter_y1 = max(y1, y1r)
        inter_x2 = min(x2, x2r)
        inter_y2 = min(y2, y2r)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2r - x1r) * (y2r - y1r)
        union = area1 + area2 - inter_area
        return inter_area / union if union > 0 else 0.0

    @staticmethod
    def _bbox(det):
        if det.bbox is None:
            return None
        cx = det.bbox.center.position.x
        cy = det.bbox.center.position.y
        w = det.bbox.size_x
        h = det.bbox.size_y
        return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)

    @staticmethod
    def _cls(det):
        return det.results[0].hypothesis.class_id if det.results else None

    @staticmethod
    def _conf(det):
        return float(det.results[0].hypothesis.score) if det.results and det.results[0].hypothesis.score is not None else None


def main():
    rclpy.init()
    node = StereoBoxDepthNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()