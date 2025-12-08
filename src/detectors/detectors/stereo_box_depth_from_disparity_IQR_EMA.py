#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from vision_msgs.msg import Detection2DArray
from stereo_msgs.msg import DisparityImage
from std_msgs.msg import String

from cv_bridge import CvBridge
import numpy as np
import json
import time
from collections import deque

# === Camera Parameters (same as before) ===
FOCAL_PX   = 555.78     # pixels
BASELINE_M = 0.0717     # meters

# === Basic depth/disparity filters ===
CONF_THRESH     = 0.35
MIN_DISPARITY   = 3.0    # px
DEPTH_MIN       = 0.3    # m
DEPTH_MAX       = 50.0   # m
MIN_VALID_PIX   = 50     # min valid disparity pixels in ROI

# === Temporal smoothing (EMA + IQR over time) ===
BUFFER_LEN          = 20       # history length per object
EMA_ALPHA           = 0.1       # EMA factor
EMA_UPDATE_EVERY    = 3         # update EMA every N detections for that object
OBJECT_EXPIRATION_S = 5.0       # forget object if not seen for this many seconds

DEBUG = False  # set True if you want detailed logs


class StereoBoxDepthIQR_EMA(Node):
    def __init__(self):
        super().__init__('stereo_box_depth_from_disparity_IQR_EMA', namespace='follower_robot')

        self.bridge = CvBridge()

        detections_topic = '/follower_robot/depth_cam/left/detections'
        disparity_topic  = '/follower_robot/depth_cam/disparity'
        out_topic        = '/follower_robot/obstacles_depth'

        self.get_logger().info(f'Listening for detections on: {detections_topic}')
        self.get_logger().info(f'Listening for disparity on:  {disparity_topic}')
        self.get_logger().info(f'Publishing depths on:        {out_topic}')

        self.det_sub = self.create_subscription(
            Detection2DArray,
            detections_topic,
            self.detections_cb,
            10
        )

        self.disp_sub = self.create_subscription(
            DisparityImage,
            disparity_topic,
            self.disparity_cb,
            10
        )

        self.depth_pub = self.create_publisher(
            String,
            out_topic,
            10
        )

        self.latest_disp = None
        self.latest_disp_stamp = None

        # Per-object smoothing state: key -> {smoothed_disp, buffer, update_count, last_seen}
        self.objects = {}

        self.get_logger().info('StereoBoxDepthFromDisparity + EMA + IQR node started.')

    # --------- Callbacks ----------

    def disparity_cb(self, msg: DisparityImage):
        """Store the latest disparity image as float32 numpy array."""
        try:
            disp = self.bridge.imgmsg_to_cv2(msg.image, desired_encoding='32FC1')
            self.latest_disp = disp
            self.latest_disp_stamp = msg.header.stamp
            if DEBUG:
                h, w = disp.shape[:2]
                self.get_logger().info(
                    f'[disp_cb] Got disparity image {w}x{h}, encoding={msg.image.encoding}'
                )
        except Exception as e:
            self.get_logger().warn(f'[disp_cb] Failed to convert disparity image: {e}')

    def detections_cb(self, msg: Detection2DArray):
        """For each detection, compute mean disparity in bbox, then apply IQR+EMA over time."""
        if self.latest_disp is None:
            if DEBUG:
                self.get_logger().warn('[det_cb] No disparity image received yet, skipping detections')
            return

        current_time = time.time()

        # Clean up old objects first
        expired = [k for k, v in self.objects.items()
                   if current_time - v['last_seen'] > OBJECT_EXPIRATION_S]
        for k in expired:
            if DEBUG:
                self.get_logger().info(f'[det_cb] Expiring object key={k}')
            del self.objects[k]

        if DEBUG:
            self.get_logger().info(f'[det_cb] Received Detection2DArray with {len(msg.detections)} detections')

        results = []

        for i, det in enumerate(msg.detections):
            bbox = self._bbox(det)
            cls_id = self._cls(det)
            conf = self._conf(det)
            track_id = det.id if det.id else None

            if DEBUG:
                self.get_logger().info(
                    f'[det_cb] det #{i}: cls={cls_id}, conf={conf}, id={track_id}, bbox={bbox}'
                )

            if bbox is None or cls_id is None:
                continue
            if conf is not None and conf < CONF_THRESH:
                continue

            # 1) Raw measurement: mean disparity over valid pixels in bbox
            raw_disp = self._mean_disparity_in_bbox(bbox)
            if raw_disp is None:
                if DEBUG:
                    self.get_logger().info(f'[det_cb] det #{i}: no valid disparity in ROI')
                continue

            if raw_disp < MIN_DISPARITY:
                if DEBUG:
                    self.get_logger().info(f'[det_cb] det #{i}: raw_disp={raw_disp:.3f} < MIN_DISPARITY')
                continue

            # 2) Per-object key (prefer track_id, else fall back to class string)
            obj_key = str(track_id) if track_id is not None else f'class_{cls_id}'

            # 3) Get/create object state
            state = self.objects.get(obj_key)
            if state is None:
                state = {
                    'smoothed_disp': None,
                    'buffer': deque(maxlen=BUFFER_LEN),
                    'update_count': 0,
                    'last_seen': current_time
                }
                self.objects[obj_key] = state

            # 4) Update timestamp
            state['last_seen'] = current_time

            # 5) Append raw measurement to temporal buffer
            state['buffer'].append(raw_disp)

            # 6) IQR filter over time (buffer)
            filtered_disp = self._iqr_filter(list(state['buffer']))

            # If IQR throws everything out (e.g. very few samples), fall back to raw
            current_disp = filtered_disp if filtered_disp is not None else raw_disp

            # 7) EMA update every N samples
            state['update_count'] += 1
            if state['update_count'] % EMA_UPDATE_EVERY == 0:
                if state['smoothed_disp'] is None:
                    state['smoothed_disp'] = current_disp
                else:
                    state['smoothed_disp'] = (
                        EMA_ALPHA * current_disp +
                        (1.0 - EMA_ALPHA) * state['smoothed_disp']
                    )

            # 8) Final disparity = EMA if available, else current filtered/raw
            final_disp = state['smoothed_disp'] if state['smoothed_disp'] is not None else current_disp

            if final_disp < MIN_DISPARITY:
                continue

            depth = FOCAL_PX * BASELINE_M / final_disp
            if not (DEPTH_MIN <= depth <= DEPTH_MAX):
                continue

            results.append({
                'id': track_id,
                'class': cls_id,
                'disparity_px': round(float(final_disp), 3),
                'depth_m': round(float(depth), 3),
                'raw_disp': round(float(raw_disp), 3)
            })

        if results:
            out_msg = String()
            out_msg.data = json.dumps(results, separators=(',', ':'))
            self.depth_pub.publish(out_msg)
            self.get_logger().info(f'object depths: {out_msg.data}')
        else:
            if DEBUG:
                self.get_logger().info('[det_cb] No valid objects this frame')

    # --------- Helpers ----------

    def _mean_disparity_in_bbox(self, bbox):
        """
        Take the disparity image and compute mean disparity inside bbox.
        Optionally shrink ROI a bit to avoid bbox borders.
        """
        if self.latest_disp is None:
            return None

        h, w = self.latest_disp.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)

        # Clamp to image
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))

        if x2 <= x1 or y2 <= y1:
            return None

        # Crop a bit inward to avoid edges (20% margin each side)
        margin_x = int(0.2 * (x2 - x1))
        margin_y = int(0.2 * (y2 - y1))
        x1i = x1 + margin_x
        x2i = x2 - margin_x
        y1i = y1 + margin_y
        y2i = y2 - margin_y

        if x2i <= x1i or y2i <= y1i:
            return None

        roi = self.latest_disp[y1i:y2i, x1i:x2i]
        if roi.size == 0:
            return None

        # Valid disparities: > 0 and finite
        valid = roi[np.isfinite(roi) & (roi > 0.0)]
        if valid.size < MIN_VALID_PIX:
            if DEBUG:
                self.get_logger().info(
                    f'[mean_disp] ROI size={roi.size}, valid={valid.size} < MIN_VALID_PIX={MIN_VALID_PIX}'
                )
            return None

        return float(valid.mean())

    @staticmethod
    def _iqr_filter(values):
        """IQR-based outlier rejection over a 1D list of values."""
        if len(values) < 3:
            return None
        arr = np.array(values, dtype=np.float32)
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            # all same or near-same, just return mean
            return float(arr.mean())

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        filtered = arr[(arr >= lower) & (arr <= upper)]
        if filtered.size == 0:
            return None
        return float(filtered.mean())

    @staticmethod
    def _bbox(det):
        """Return (x1, y1, x2, y2) from Detection2D bbox."""
        if det.bbox is None:
            return None
        cx = det.bbox.center.position.x
        cy = det.bbox.center.position.y
        w = det.bbox.size_x
        h = det.bbox.size_y
        return (cx - w / 2.0, cy - h / 2.0,
                cx + w / 2.0, cy + h / 2.0)

    @staticmethod
    def _cls(det):
        return det.results[0].hypothesis.class_id if det.results else None

    @staticmethod
    def _conf(det):
        if det.results and det.results[0].hypothesis.score is not None:
            return float(det.results[0].hypothesis.score)
        return None


def main():
    rclpy.init()
    node = StereoBoxDepthIQR_EMA()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
