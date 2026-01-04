#!/usr/bin/env python3
import math
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from stereo_msgs.msg import DisparityImage
from std_msgs.msg import Float32
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

from detectors_msgs.msg import GoalMarkerState

# Stereo depth params
FOCAL_PX = 555.78
BASELINE_M = 0.0717

# Left camera intrinsics (for bearing only)
FX_LEFT = 555.09075
CX_LEFT = 319.05936

# Disparity / ROI tuning
MIN_VALID_PIX = 50          # min valid disparity pixels in ROI
MIN_DISPARITY = 0.5         # px, below this depth is unreliable
DEPTH_MIN = 0.2             # m
DEPTH_MAX = 50.0            # m

# Set PRINT=True while validating depth estimates. Set to False for silent operation.
PRINT = False


class GoalMarkerDepth(Node):
    def __init__(self):
        super().__init__("goal_marker_depth", namespace="follower_robot")

        # --- Parameters ---
        self.declare_parameter("left_topic", "/follower_robot/depth_cam/left/image_rect")
        self.declare_parameter("disparity_topic", "/follower_robot/depth_cam/disparity")
        self.declare_parameter("slop", 0.05)
        self.declare_parameter("min_area", 500.0)
        self.declare_parameter("depth_window", 5)

        # Performance/robustness knobs
        self.declare_parameter("det_downscale", 0.5)     # 0.5 = detect on half-res
        self.declare_parameter("track_margin", 0.5)      # expand last bbox by 50% each side
        self.declare_parameter("max_lost", 5)            # frames to keep tracking before full search

        left_topic = self.get_parameter("left_topic").get_parameter_value().string_value
        disparity_topic = (
            self.get_parameter("disparity_topic").get_parameter_value().string_value
        )
        slop = self.get_parameter("slop").get_parameter_value().double_value

        self.min_area = self.get_parameter("min_area").get_parameter_value().double_value

        depth_window_param = (
            self.get_parameter("depth_window").get_parameter_value().integer_value
        )
        if depth_window_param < 1:
            self.get_logger().warn(f"depth_window={depth_window_param} is invalid; using 1.")
            depth_window_param = 1
        self.depth_window_size = depth_window_param
        self.depth_history = deque(maxlen=self.depth_window_size)

        self.det_downscale = float(
            self.get_parameter("det_downscale").get_parameter_value().double_value
        )
        if not (0.1 <= self.det_downscale <= 1.0):
            self.get_logger().warn(
                f"det_downscale={self.det_downscale} out of [0.1, 1.0]; using 0.5."
            )
            self.det_downscale = 0.5

        self.track_margin = float(
            self.get_parameter("track_margin").get_parameter_value().double_value
        )
        self.max_lost = int(
            self.get_parameter("max_lost").get_parameter_value().integer_value
        )
        self.max_lost = max(0, self.max_lost)

        self.bridge = CvBridge()

        # Precompute HSV thresholds & morphology kernel (avoid per-frame allocations)
        self._lower1 = np.array([0, 80, 80], dtype=np.uint8)
        self._upper1 = np.array([10, 255, 255], dtype=np.uint8)
        self._lower2 = np.array([170, 80, 80], dtype=np.uint8)
        self._upper2 = np.array([180, 255, 255], dtype=np.uint8)
        self._kernel = np.ones((10, 10), np.uint8)

        # Simple tracker state (bbox in full-res image coords)
        self._track_bbox: Optional[Tuple[int, int, int, int]] = None
        self._lost_count: int = 0

        # --- Subscribers: LEFT image + disparity (approx sync) ---
        self.left_sub = Subscriber(self, Image, left_topic, qos_profile=10)
        self.disp_sub = Subscriber(self, DisparityImage, disparity_topic, qos_profile=10)

        self.sync = ApproximateTimeSynchronizer(
            [self.left_sub, self.disp_sub],
            queue_size=10,
            slop=slop,
        )
        self.sync.registerCallback(self.cb_pair)

        # --- Publishers ---
        self.depth_pub = self.create_publisher(
            Float32, "/follower_robot/depth_cam/goal_marker_depth", 10
        )
        self.depth_raw_pub = self.create_publisher(
            Float32, "/follower_robot/depth_cam/goal_marker_depth_raw", 10
        )

        self.state_pub = self.create_publisher(
            GoalMarkerState,
            "/follower_robot/depth_cam/goal_marker_state",
            10,
        )

        self.get_logger().info(
            f"GoalMarkerDepth ready: red goal detection on LEFT + depth from disparity ROI "
            f"(MA window={self.depth_window_size}, det_downscale={self.det_downscale})"
        )

    # ---- Synced callback: LEFT image + disparity ----
    def cb_pair(self, left_msg: Image, disp_msg: DisparityImage):
        # Convert inputs
        try:
            left_bgr = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge LEFT conversion failed: {e}")
            return

        try:
            disp = self.bridge.imgmsg_to_cv2(disp_msg.image, desired_encoding="32FC1")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge disparity conversion failed: {e}")
            return

        # Prepare default state (NOT visible)
        state = GoalMarkerState()
        state.header = left_msg.header
        state.visible = False
        state.depth_m = 0.0
        state.bearing_rad = 0.0
        state.cx = -1.0
        state.cy = -1.0

        # Detect bbox in LEFT
        bbox = self._detect_red_bbox_tracked(left_bgr)
        if bbox is None:
            self.state_pub.publish(state)
            return

        # Mean disparity within bbox ROI
        disparity = self._mean_disparity_in_bbox(disp, bbox)
        if disparity is None or disparity < MIN_DISPARITY:
            self.state_pub.publish(state)
            if PRINT:
                self.get_logger().info("Goal detected but disparity invalid/too small.")
            return

        depth_m_raw = (FOCAL_PX * BASELINE_M) / float(disparity)
        if not (DEPTH_MIN <= depth_m_raw <= DEPTH_MAX):
            self.state_pub.publish(state)
            if PRINT:
                self.get_logger().info(
                    f"Depth out of range: {depth_m_raw:.3f} m (disp={disparity:.3f}px)"
                )
            return

        # Smooth depth
        self.depth_history.append(float(depth_m_raw))
        depth_m_smooth = float(sum(self.depth_history) / len(self.depth_history))

        # Center pixel (LEFT)
        x, y, w, h = bbox
        cx_px = x + w / 2.0
        cy_px = y + h / 2.0

        bearing_rad = math.atan2((cx_px - CX_LEFT), FX_LEFT)

        # Publish legacy numeric depths
        msg_smooth = Float32()
        msg_smooth.data = depth_m_smooth
        self.depth_pub.publish(msg_smooth)

        msg_raw = Float32()
        msg_raw.data = float(depth_m_raw)
        self.depth_raw_pub.publish(msg_raw)

        # Publish state
        state.visible = True
        state.depth_m = float(depth_m_smooth)
        state.bearing_rad = float(bearing_rad)
        state.cx = float(cx_px)
        state.cy = float(cy_px)
        self.state_pub.publish(state)

        if PRINT:
            self.get_logger().info(
                f"Goal depth: smooth={depth_m_smooth:.3f} m | raw={depth_m_raw:.3f} m "
                f"(disp={disparity:.2f}px) | bearing={bearing_rad:.3f} rad | "
                f"bbox=({x},{y},{w},{h})"
            )

    # ---- Tracking-aware detection (fast path most frames) ----
    def _detect_red_bbox_tracked(self, bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        h, w = bgr.shape[:2]

        # Try local search around last bbox
        if self._track_bbox is not None and self._lost_count < self.max_lost:
            x, y, bw, bh = self._track_bbox
            mx = int(max(5, bw * self.track_margin))
            my = int(max(5, bh * self.track_margin))

            x1 = max(0, x - mx)
            y1 = max(0, y - my)
            x2 = min(w, x + bw + mx)
            y2 = min(h, y + bh + my)

            crop = bgr[y1:y2, x1:x2]
            local = self._detect_red_bbox(crop)
            if local is not None:
                lx, ly, lw, lh = local
                bbox = (x1 + lx, y1 + ly, lw, lh)
                self._track_bbox = bbox
                self._lost_count = 0
                return bbox

            self._lost_count += 1

        # Full-frame search
        bbox = self._detect_red_bbox(bgr)
        if bbox is not None:
            self._track_bbox = bbox
            self._lost_count = 0
            return bbox

        # Lost
        self._track_bbox = None
        self._lost_count = 0
        return None

    # ---- Red detection on an image (or crop) ----
    def _detect_red_bbox(self, bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        if bgr.size == 0:
            return None

        # Downscale for speed
        scale = self.det_downscale
        if scale < 1.0:
            small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            small = bgr

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, self._lower1, self._upper1)
        mask2 = cv2.inRange(hsv, self._lower2, self._upper2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Morphology to clean noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=1)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        cnt = max(cnts, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        min_area_scaled = self.min_area * (scale * scale)
        if area < min_area_scaled:
            return None

        x, y, w, h = cv2.boundingRect(cnt)

        # Scale bbox back to full-res coordinates (of the input 'bgr')
        if scale < 1.0:
            inv = 1.0 / scale
            x = int(round(x * inv))
            y = int(round(y * inv))
            w = int(round(w * inv))
            h = int(round(h * inv))

        return (x, y, w, h)

    # ---- Mean disparity inside bbox on disparity image ----
    def _mean_disparity_in_bbox(
        self, disp: np.ndarray, rect: Tuple[int, int, int, int]
    ) -> Optional[float]:
        x, y, w, h = rect
        h_img, w_img = disp.shape[:2]

        x1 = max(0, min(w_img - 1, int(x)))
        y1 = max(0, min(h_img - 1, int(y)))
        x2 = max(0, min(w_img, int(x + w)))
        y2 = max(0, min(h_img, int(y + h)))

        if x2 <= x1 or y2 <= y1:
            return None

        roi = disp[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        valid = np.isfinite(roi) & (roi > 0.0)
        n_valid = int(valid.sum())
        if n_valid < MIN_VALID_PIX:
            return None

        # cv2.mean avoids fancy-index copy; mask must be uint8 (0 or 255)
        mask = (valid.astype(np.uint8) * 255)
        mean_val = cv2.mean(roi, mask=mask)[0]
        return float(mean_val)


def main():
    rclpy.init()
    node = GoalMarkerDepth()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
