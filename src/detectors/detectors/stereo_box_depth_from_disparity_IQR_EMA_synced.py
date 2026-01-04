#!/usr/bin/env python3
"""
StereoBoxDepthFromDisparity with IQR-over-time + EMA smoothing (synced + safer object keys)

Changes vs stereo_box_depth_from_disparity_IQR_EMA.py:
- Proper ApproximateTimeSynchronizer on (Detection2DArray, DisparityImage)
- Faster ROI stats: mask + cv2.mean (avoids fancy-index copies)
- Adaptive minimum-valid pixels (ratio + floor), better for small/far boxes
- Safer per-object keying when det.id is missing (class + quantized center)
- Optional processing rate limit (max_process_hz)
- print_json parameter so you can validate then silence logs (runtime)
"""
import json
import time
from collections import deque
from typing import Optional, Tuple, Dict, Any, List
import math


import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from vision_msgs.msg import Detection2DArray
from stereo_msgs.msg import DisparityImage
from std_msgs.msg import String
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

# === Camera Parameters ===
FOCAL_PX   = 555.78
BASELINE_M = 0.0717

# --- Camera geometry ---
IMAGE_WIDTH_PX = 640.0     # must match your camera config
CX_PX = IMAGE_WIDTH_PX * 0.5

# --- Camera mounting relative to base_link ---
CAMERA_X_M = 0.35
CAMERA_Y_M = 0.0

DEBUG = False


class StereoBoxDepthIQR_EMA_Synced(Node):
    def __init__(self):
        super().__init__('stereo_box_depth_from_disparity_IQR_EMA_synced', namespace='follower_robot')
        self.bridge = CvBridge()

        # Topics
        self.declare_parameter('detections_topic', '/follower_robot/depth_cam/left/detections')
        self.declare_parameter('disparity_topic',  '/follower_robot/depth_cam/disparity')
        self.declare_parameter('out_topic','/follower_robot/obstacles_depth')

        # Matching: images ~10 Hz, disparity ~8 Hz
        self.declare_parameter('sync_queue', 10)
        self.declare_parameter('sync_slop', 0.12)

        # Filters
        self.declare_parameter('conf_thresh', 0.35)
        self.declare_parameter('min_disparity', 3.0)
        self.declare_parameter('depth_min', 0.3)
        self.declare_parameter('depth_max', 50.0)

        # ROI / validity
        self.declare_parameter('roi_shrink', 0.2)
        self.declare_parameter('min_valid_ratio', 0.05)
        self.declare_parameter('min_valid_floor', 10)

        # Temporal smoothing
        self.declare_parameter('buffer_len', 20)
        self.declare_parameter('ema_alpha', 0.10)
        self.declare_parameter('ema_update_every', 3)
        self.declare_parameter('object_expiration_s', 5.0)

        # Keying when det.id is missing
        self.declare_parameter('key_bin_px', 40)  # quantization bin for bbox center (pixels)

        # Performance & logging
        self.declare_parameter('max_process_hz', 0.0)
        self.declare_parameter('class_allowlist', [])
        self.declare_parameter('print_json', True)

        # Load params
        self.detections_topic = self.get_parameter('detections_topic').value
        self.disparity_topic  = self.get_parameter('disparity_topic').value
        self.out_topic        = self.get_parameter('out_topic').value

        self.sync_queue = int(self.get_parameter('sync_queue').value)
        self.sync_slop  = float(self.get_parameter('sync_slop').value)

        self.conf_thresh   = float(self.get_parameter('conf_thresh').value)
        self.min_disparity = float(self.get_parameter('min_disparity').value)
        self.depth_min     = float(self.get_parameter('depth_min').value)
        self.depth_max     = float(self.get_parameter('depth_max').value)

        self.roi_shrink      = float(self.get_parameter('roi_shrink').value)
        self.min_valid_ratio = float(self.get_parameter('min_valid_ratio').value)
        self.min_valid_floor = int(self.get_parameter('min_valid_floor').value)

        self.buffer_len          = max(3, int(self.get_parameter('buffer_len').value))
        self.ema_alpha           = float(self.get_parameter('ema_alpha').value)
        self.ema_update_every    = max(1, int(self.get_parameter('ema_update_every').value))
        self.object_expiration_s = float(self.get_parameter('object_expiration_s').value)

        self.key_bin_px = max(10, int(self.get_parameter('key_bin_px').value))

        self.max_process_hz = float(self.get_parameter('max_process_hz').value)
        self.allowlist: List[str] = [str(x) for x in self.get_parameter('class_allowlist').value]
        self.print_json = bool(self.get_parameter('print_json').value)

        self._last_process_t = 0.0

        # Per-object smoothing state
        self.objects: Dict[str, Dict[str, Any]] = {}

        self.get_logger().info(f'Listening for detections on: {self.detections_topic}')
        self.get_logger().info(f'Listening for disparity on:  {self.disparity_topic}')
        self.get_logger().info(f'Publishing depths on:        {self.out_topic}')
        self.get_logger().info(f'Sync slop={self.sync_slop}s queue={self.sync_queue} (img~10Hz disp~8Hz)')

        # Synced subs
        self.det_sub  = Subscriber(self, Detection2DArray, self.detections_topic, qos_profile=10)
        self.disp_sub = Subscriber(self, DisparityImage, self.disparity_topic, qos_profile=10)
        self.sync = ApproximateTimeSynchronizer(
            [self.det_sub, self.disp_sub],
            queue_size=self.sync_queue,
            slop=self.sync_slop,
        )
        self.sync.registerCallback(self.synced_cb)

        self.depth_pub = self.create_publisher(String, self.out_topic, 10)
        self.get_logger().info('StereoBoxDepthIQR_EMA_Synced node started.')

    def synced_cb(self, det_msg: Detection2DArray, disp_msg: DisparityImage):
        # Optional rate limit
        if self.max_process_hz and self.max_process_hz > 0.0:
            now = time.time()
            if (now - self._last_process_t) < (1.0 / self.max_process_hz):
                return
            self._last_process_t = now

        # Convert disparity once
        try:
            disp = self.bridge.imgmsg_to_cv2(disp_msg.image, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().warn(f'[synced_cb] Failed to convert disparity image: {e}')
            return

        current_time = time.time()
        self._expire_objects(current_time)

        results = []

        for det in det_msg.detections:
            bbox = self._bbox(det)
            cls_id = self._cls(det)
            conf = self._conf(det)
            track_id = det.id if det.id else None

            if bbox is None or cls_id is None:
                continue

            cls_str = str(cls_id)
            if self.allowlist and cls_str not in self.allowlist:
                continue
            if conf is not None and conf < self.conf_thresh:
                continue

            raw_disp = self._mean_disparity_in_bbox(disp, bbox)
            if raw_disp is None or raw_disp < self.min_disparity:
                continue

            key = self._object_key(track_id, cls_str, bbox)

            state = self.objects.get(key)
            if state is None:
                state = {
                    'smoothed_disp': None,
                    'buffer': deque(maxlen=self.buffer_len),
                    'update_count': 0,
                    'last_seen': current_time
                }
                self.objects[key] = state

            state['last_seen'] = current_time
            state['buffer'].append(float(raw_disp))

            filtered_disp = self._iqr_filter(list(state['buffer']))
            current_disp = filtered_disp if filtered_disp is not None else float(raw_disp)

            state['update_count'] += 1
            if self.ema_update_every <= 1 or (state['update_count'] % self.ema_update_every == 0):
                if state['smoothed_disp'] is None:
                    state['smoothed_disp'] = current_disp
                else:
                    state['smoothed_disp'] = (
                        self.ema_alpha * current_disp +
                        (1.0 - self.ema_alpha) * state['smoothed_disp']
                    )

            final_disp = state['smoothed_disp'] if state['smoothed_disp'] is not None else current_disp
            if final_disp < self.min_disparity:
                continue

            depth = FOCAL_PX * BASELINE_M / final_disp
            if not (self.depth_min <= depth <= self.depth_max):
                continue

            # --- NEW: estimate obstacle XY in base_link frame (approx) ---
            # bbox center pixel
            cx_box = 0.5 * (bbox[0] + bbox[2])  # pixels

            # bearing in camera frame (approx, horizontal only)
            bearing = math.atan2((cx_box - CX_PX), FOCAL_PX)

            # position relative to camera in XY plane
            x_cam_rel = float(depth) * math.cos(bearing)
            y_cam_rel = float(depth) * math.sin(bearing)

            # transform into base_link frame using your known camera offset
            x_base = CAMERA_X_M + x_cam_rel
            y_base = CAMERA_Y_M + y_cam_rel

            results.append({
                'id': track_id,
                'class': cls_id,
                'disparity_px': round(float(final_disp), 3),
                'depth_m': round(float(depth), 3),
                'raw_disp': round(float(raw_disp), 3),

                # --- NEW fields ---
                'bearing_rad': round(float(bearing), 4),
                'x_m': round(float(x_base), 3),
                'y_m': round(float(y_base), 3),
            })


        if results:
            out_msg = String()
            out_msg.data = json.dumps(results, separators=(',', ':'))
            self.depth_pub.publish(out_msg)
            if self.print_json:
                self.get_logger().info(f'object depths: {out_msg.data}')
        else:
            if DEBUG:
                self.get_logger().info('[synced_cb] No valid objects this frame')

    def _expire_objects(self, now_s: float):
        expired = [k for k, v in self.objects.items()
                   if now_s - float(v['last_seen']) > self.object_expiration_s]
        for k in expired:
            del self.objects[k]

    def _mean_disparity_in_bbox(self, disp: np.ndarray, bbox) -> Optional[float]:
        h, w = disp.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)

        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return None

        # Center crop
        if self.roi_shrink > 0.0:
            mx = int(self.roi_shrink * (x2 - x1))
            my = int(self.roi_shrink * (y2 - y1))
            x1 += mx; x2 -= mx
            y1 += my; y2 -= my
            if x2 <= x1 or y2 <= y1:
                return None

        roi = disp[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        valid = np.isfinite(roi) & (roi > 0.0)
        n_valid = int(valid.sum())

        min_valid = max(self.min_valid_floor, int(self.min_valid_ratio * roi.size))
        if n_valid < min_valid:
            return None

        mask = (valid.astype(np.uint8) * 255)
        mean_val = cv2.mean(roi, mask=mask)[0]
        return float(mean_val)

    @staticmethod
    def _iqr_filter(values):
        if len(values) < 3:
            return None
        arr = np.array(values, dtype=np.float32)
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            return float(arr.mean())
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        filtered = arr[(arr >= lower) & (arr <= upper)]
        if filtered.size == 0:
            return None
        return float(filtered.mean())

    def _object_key(self, track_id, cls_str: str, bbox) -> str:
        if track_id is not None:
            return str(track_id)
        cx = 0.5 * (bbox[0] + bbox[2])
        cy = 0.5 * (bbox[1] + bbox[3])
        b = self.key_bin_px
        qx = int(cx // b)
        qy = int(cy // b)
        return f'{cls_str}:{qx}:{qy}'

    @staticmethod
    def _bbox(det) -> Optional[Tuple[float, float, float, float]]:
        if det.bbox is None:
            return None
        cx = det.bbox.center.position.x
        cy = det.bbox.center.position.y
        bw = det.bbox.size_x
        bh = det.bbox.size_y
        return (cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0)

    @staticmethod
    def _cls(det):
        return det.results[0].hypothesis.class_id if det.results else None

    @staticmethod
    def _conf(det) -> Optional[float]:
        if det.results and det.results[0].hypothesis.score is not None:
            return float(det.results[0].hypothesis.score)
        return None


def main():
    rclpy.init()
    node = StereoBoxDepthIQR_EMA_Synced()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
