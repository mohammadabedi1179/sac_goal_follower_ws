#!/usr/bin/env python3
import math
from collections import deque

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

# Stereo depth params (your previous setup)
FOCAL_PX = 555.78
BASELINE_M = 0.0717

# Left camera intrinsics (from your calibration / old goal_env.py)
FX_LEFT = 1360.02116
CX_LEFT = 820.01429

# Disparity / ROI tuning
MIN_VALID_PIX = 50          # min valid disparity pixels in ROI
MIN_DISPARITY = 0.5         # px, below this depth is unreliable
DEPTH_MIN = 0.2             # m
DEPTH_MAX = 50.0            # m
MAX_DISP_DT = 0.05          # max allowed time diff between left image and disparity (sec)


class GoalMarkerDepth(Node):
    def __init__(self):
        super().__init__("goal_marker_depth", namespace="follower_robot")

        # --- Parameters (same names as before, with sensible defaults) ---
        self.declare_parameter(
            "left_topic",
            "/follower_robot/depth_cam/left/image_rect",
        )
        self.declare_parameter(
            "right_topic",
            "/follower_robot/depth_cam/right/image_rect",
        )
        self.declare_parameter("slop", 0.03)
        self.declare_parameter("min_area", 300.0)
        self.declare_parameter("show_debug", False)
        self.declare_parameter("publish_overlay", True)
        self.declare_parameter("depth_window", 5)
        self.declare_parameter(
            "disparity_topic",
            "/follower_robot/depth_cam/disparity",
        )

        left_topic = (
            self.get_parameter("left_topic").get_parameter_value().string_value
        )
        right_topic = (
            self.get_parameter("right_topic").get_parameter_value().string_value
        )
        slop = self.get_parameter("slop").get_parameter_value().double_value
        self.min_area = (
            self.get_parameter("min_area").get_parameter_value().double_value
        )
        self.show_debug = (
            self.get_parameter("show_debug").get_parameter_value().bool_value
        )
        self.publish_overlay = (
            self.get_parameter("publish_overlay").get_parameter_value().bool_value
        )
        disparity_topic = (
            self.get_parameter("disparity_topic").get_parameter_value().string_value
        )

        depth_window_param = (
            self.get_parameter("depth_window")
            .get_parameter_value()
            .integer_value
        )
        if depth_window_param < 1:
            self.get_logger().warn(
                f"depth_window={depth_window_param} is invalid; using 1."
            )
            depth_window_param = 1
        self.depth_window_size = depth_window_param
        self.depth_history = deque(maxlen=self.depth_window_size)

        self.bridge = CvBridge()

        # --- Subscribers: stereo pair with approximate sync (for red detection & overlays) ---
        self.left_sub = Subscriber(self, Image, left_topic, qos_profile=10)
        self.right_sub = Subscriber(self, Image, right_topic, qos_profile=10)

        self.sync = ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub],
            queue_size=10,
            slop=slop,
        )
        self.sync.registerCallback(self.cb_pair)

        # --- Disparity subscriber (from stereo_image_proc) ---
        self.disp_sub = self.create_subscription(
            DisparityImage,
            disparity_topic,
            self.disparity_cb,
            10,
        )
        self.latest_disp = None
        self.latest_disp_stamp = None

        # --- Publishers ---
        # Numeric depths (backward-compatible with your current setup)
        self.depth_pub = self.create_publisher(
            Float32, "/follower_robot/depth_cam/goal_marker_depth", 10
        )
        self.depth_raw_pub = self.create_publisher(
            Float32, "/follower_robot/depth_cam/goal_marker_depth_raw", 10
        )

        # Overlays
        if self.publish_overlay:
            self.left_overlay_pub = self.create_publisher(
                Image,
                "/follower_robot/depth_cam/left/goal_overlay",
                10,
            )
            self.right_overlay_pub = self.create_publisher(
                Image,
                "/follower_robot/depth_cam/right/goal_overlay",
                10,
            )
        else:
            self.left_overlay_pub = None
            self.right_overlay_pub = None

        # High-level state publisher for RL env
        self.state_pub = self.create_publisher(
            GoalMarkerState,
            "/follower_robot/depth_cam/goal_marker_state",
            10,
        )

        self.get_logger().info(
            f"GoalMarkerDepth node ready: red-cube detection + depth from disparity ROI "
            f"(MA window={self.depth_window_size})"
        )

    # ---- Disparity callback ----
    def disparity_cb(self, msg: DisparityImage):
        """Store the latest disparity image (32FC1) and timestamp."""
        try:
            disp = self.bridge.imgmsg_to_cv2(msg.image, desired_encoding="32FC1")
            self.latest_disp = disp
            self.latest_disp_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().warn(f"disparity_cb: cv_bridge conversion failed: {e}")

    # ---- Stereo callback (left/right images) ----
    def cb_pair(self, left_msg: Image, right_msg: Image):
        # Convert to BGR
        try:
            left_bgr = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding="bgr8")
            right_bgr = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        # Detect red marker in both images
        L, l_rect, l_overlay = self.find_red_marker(left_bgr)
        R, r_rect, r_overlay = self.find_red_marker(right_bgr)

        # Publish overlays (optional)
        if self.publish_overlay:
            try:
                if self.left_overlay_pub is not None:
                    self.left_overlay_pub.publish(
                        self.bridge.cv2_to_imgmsg(l_overlay, encoding="bgr8")
                    )
                if self.right_overlay_pub is not None:
                    self.right_overlay_pub.publish(
                        self.bridge.cv2_to_imgmsg(r_overlay, encoding="bgr8")
                    )
            except Exception as e:
                self.get_logger().warn(f"Overlay publish failed: {e}")

        # Prepare default state (NOT visible)
        state = GoalMarkerState()
        state.header = left_msg.header
        state.visible = False
        state.depth_m = 0.0
        state.bearing_rad = 0.0
        state.cx = -1.0
        state.cy = -1.0

        # ---- CASE 1: no detection on either side ----
        if L is None or R is None:
            self.state_pub.publish(state)
            if self.show_debug:
                self.get_logger().info("Goal marker not detected.")
            return

        # ---- Ensure we have a disparity image, and it's near-synchronized ----
        if self.latest_disp is None or self.latest_disp_stamp is None:
            self.state_pub.publish(state)
            if self.show_debug:
                self.get_logger().info("No disparity image yet, skipping depth.")
            return

        # Simple time sync check
        lt = left_msg.header.stamp.sec + left_msg.header.stamp.nanosec * 1e-9
        dt = abs(
            lt
            - (
                self.latest_disp_stamp.sec
                + self.latest_disp_stamp.nanosec * 1e-9
            )
        )
        if dt > MAX_DISP_DT:
            # Too old / too far: skip this frame
            self.state_pub.publish(state)
            if self.show_debug:
                self.get_logger().info(
                    f"Disparity too far in time (dt={dt:.3f}s > {MAX_DISP_DT}), skipping."
                )
            return

        # ---- Compute mean disparity inside LEFT bbox ----
        disparity = self.mean_disparity_in_bbox(L)
        if disparity is None or disparity <= MIN_DISPARITY:
            self.state_pub.publish(state)
            if self.show_debug:
                self.get_logger().info(
                    f"Invalid or too small disparity in ROI: {disparity}"
                )
            return

        # ---- Compute depth from disparity ----
        depth_m_raw = (FOCAL_PX * BASELINE_M) / disparity
        if not (DEPTH_MIN <= depth_m_raw <= DEPTH_MAX):
            self.state_pub.publish(state)
            if self.show_debug:
                self.get_logger().info(
                    f"Depth out of range: {depth_m_raw:.3f} m (disp={disparity:.3f})"
                )
            return

        self.depth_history.append(depth_m_raw)
        depth_m_smooth = float(sum(self.depth_history) / len(self.depth_history))

        # pixel center from LEFT rect
        lx, ly, lw, lh = L
        cx_px = lx + lw / 2.0
        cy_px = ly + lh / 2.0

        bearing_rad = math.atan2((cx_px - CX_LEFT), FX_LEFT)

        # Publish legacy numeric depths
        msg_smooth = Float32()
        msg_smooth.data = depth_m_smooth
        self.depth_pub.publish(msg_smooth)

        msg_raw = Float32()
        msg_raw.data = depth_m_raw
        self.depth_raw_pub.publish(msg_raw)

        # Fill and publish state as VISIBLE
        state.visible = True
        state.depth_m = float(depth_m_smooth)
        state.bearing_rad = float(bearing_rad)
        state.cx = float(cx_px)
        state.cy = float(cy_px)
        self.state_pub.publish(state)

        self.get_logger().info(
            f"Goal marker depth (smooth): {depth_m_smooth:.3f} m | "
            f"raw: {depth_m_raw:.3f} m (disp {disparity:.2f}px) | "
            f"bearing: {bearing_rad:.3f} rad"
        )

        if self.show_debug:
            cv2.imshow("goal_left", l_overlay)
            cv2.imshow("goal_right", r_overlay)
            cv2.waitKey(1)

    # ---- Red detection & rectangle extraction ----
    def find_red_marker(self, bgr):
        overlay = bgr.copy()

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        lower1 = np.array([0, 80, 80], dtype=np.uint8)
        upper1 = np.array([10, 255, 255], dtype=np.uint8)
        lower2 = np.array([170, 80, 80], dtype=np.uint8)
        upper2 = np.array([180, 255, 255], dtype=np.uint8)

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((10, 10), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        cnts, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not cnts:
            return None, None, overlay

        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < self.min_area:
            return None, None, overlay

        x, y, w, h = cv2.boundingRect(cnt)

        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            overlay,
            f"red goal (area={int(area)})",
            (x, max(0, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        # Return the same rect for convenience (L,R)
        return (x, y, w, h), (x, y, w, h), overlay

    # ---- Mean disparity inside the LEFT bbox ----
    def mean_disparity_in_bbox(self, rect):
        """
        rect = (x, y, w, h) in pixel coords on left rectified image.
        Uses latest disparity image (aligned with left) and returns mean
        of valid disparities in the ROI.
        """
        if self.latest_disp is None:
            return None

        x, y, w, h = rect
        h_img, w_img = self.latest_disp.shape[:2]

        x1 = max(0, min(w_img - 1, int(x)))
        y1 = max(0, min(h_img - 1, int(y)))
        x2 = max(0, min(w_img, int(x + w)))
        y2 = max(0, min(h_img, int(y + h)))

        if x2 <= x1 or y2 <= y1:
            return None

        roi = self.latest_disp[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        # Valid disparities: > 0, finite
        valid = roi[np.isfinite(roi) & (roi > 0.0)]
        if valid.size < MIN_VALID_PIX:
            return None

        # You can switch to np.median(valid) if you want extra robustness
        return float(valid.mean())


def main():
    rclpy.init()
    node = GoalMarkerDepth()
    try:
        rclpy.spin(node)
    finally:
        if node.get_parameter("show_debug").get_parameter_value().bool_value:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
