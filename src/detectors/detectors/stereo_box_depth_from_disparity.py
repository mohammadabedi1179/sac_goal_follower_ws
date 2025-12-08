#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import Image
from std_msgs.msg import String
from stereo_msgs.msg import DisparityImage


import numpy as np
import json
from cv_bridge import CvBridge

# === Camera Parameters (set from your calibration) ===
FOCAL_PX   = 555.78   # same as your other depth node
BASELINE_M = 0.0717   # 7.17 cm optical baseline

# === Tuning Parameters ===
CONF_THRESH     = 0.35
MIN_DISPARITY   = 3.0    # px, below this depth becomes huge/unreliable
DEPTH_MIN       = 0.3    # m
DEPTH_MAX       = 50.0   # m
MIN_VALID_PIX   = 50     # how many valid disparity pixels needed in ROI

DEBUG = False


class StereoBoxDepthFromDisparity(Node):
    def __init__(self):
        super().__init__('stereo_box_depth_from_disparity', namespace='follower_robot')

        self.bridge = CvBridge()

        # ---- TOPIC NAMES: change here if your YOLO/disparity topics are different ----
        detections_topic = '/follower_robot/depth_cam/left/detections'
        disparity_topic  = '/follower_robot/depth_cam/disparity'
        out_topic        = '/follower_robot/obstacles_depth'

        self.get_logger().info(f'Listening for detections on: {detections_topic}')
        self.get_logger().info(f'Listening for disparity on:  {disparity_topic}')
        self.get_logger().info(f'Publishing depths on:        {out_topic}')

        # Subscribe to YOLO detections (left camera)
        self.det_sub = self.create_subscription(
            Detection2DArray,
            detections_topic,
            self.detections_cb,
            10
        )

        # Subscribe to disparity image from stereo_image_proc
        self.disp_sub = self.create_subscription(
            DisparityImage,
            disparity_topic,
            self.disparity_cb,
            10
        )

        # Publish object depths as JSON string (same style as your other node)
        self.depth_pub = self.create_publisher(
            String,
            out_topic,
            10
        )

        self.latest_disp = None
        self.latest_disp_stamp = None

        self.get_logger().info('StereoBoxDepthFromDisparity node started.')

    # ------------- Callbacks -------------

    def disparity_cb(self, msg: DisparityImage):
        """Store the latest disparity image as float32 numpy array."""
        try:
            # msg.image is sensor_msgs/Image (usually 32FC1 disparity)
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
        """For each detection, compute mean disparity inside bbox and convert to depth."""
        if DEBUG:
            self.get_logger().info(f'[det_cb] Received Detection2DArray with {len(msg.detections)} detections')

        if self.latest_disp is None:
            if DEBUG:
                self.get_logger().warn('[det_cb] No disparity image received yet, skipping detections')
            return

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
                if DEBUG:
                    self.get_logger().warn(f'[det_cb] det #{i}: missing bbox or class, skipping')
                continue
            if conf is not None and conf < CONF_THRESH:
                if DEBUG:
                    self.get_logger().info(f'[det_cb] det #{i}: conf {conf:.3f} < {CONF_THRESH}, skipping')
                continue

            mean_disp = self._mean_disparity_in_bbox(bbox)
            if mean_disp is None:
                if DEBUG:
                    self.get_logger().info(f'[det_cb] det #{i}: no valid disparity in ROI, skipping')
                continue

            if mean_disp < MIN_DISPARITY:
                if DEBUG:
                    self.get_logger().info(
                        f'[det_cb] det #{i}: mean disparity {mean_disp:.3f} < MIN_DISPARITY, skipping'
                    )
                continue

            depth = FOCAL_PX * BASELINE_M / mean_disp

            if not (DEPTH_MIN <= depth <= DEPTH_MAX):
                if DEBUG:
                    self.get_logger().info(
                        f'[det_cb] det #{i}: depth {depth:.3f} out of [{DEPTH_MIN}, {DEPTH_MAX}], skipping'
                    )
                continue

            results.append({
                'id': track_id,
                'class': cls_id,
                'disparity_px': round(float(mean_disp), 3),
                'depth_m': round(float(depth), 3)
            })

        if results:
            out_msg = String()
            out_msg.data = json.dumps(results, separators=(',', ':'))
            self.depth_pub.publish(out_msg)
            self.get_logger().info(f'object depths: {out_msg.data}')
        else:
            if DEBUG:
                self.get_logger().info('[det_cb] No valid objects to publish in this frame')

    # ------------- Helpers -------------

    def _mean_disparity_in_bbox(self, bbox):
        """
        Take the disparity image and compute mean disparity inside bbox.
        bbox = (x1, y1, x2, y2) in pixel coordinates.
        """
        if self.latest_disp is None:
            return None

        h, w = self.latest_disp.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)

        # Clamp ROI to image bounds
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))

        if x2 <= x1 or y2 <= y1:
            return None

        roi = self.latest_disp[y1:y2, x1:x2]
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
    def _bbox(det):
        """Return (x1, y1, x2, y2) from Detection2D bbox (in pixel coords)."""
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
    node = StereoBoxDepthFromDisparity()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
