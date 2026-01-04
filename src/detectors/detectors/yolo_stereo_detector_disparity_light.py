#!/usr/bin/env python3
import time
from typing import Optional

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from cv_bridge import CvBridge

from ultralytics import YOLO

# Set True only when validating; set False for silent operation.
PRINT = False


class YoloStereoDetector(Node):
    """
    Lightweight YOLO detector for stereo pipelines.

    Key changes vs your original:
      - NO temp file / disk I/O (runs YOLO directly on numpy image)
      - Optional rate-limit (max_hz)
      - Optional left-only mode (default True) since depth-from-disparity typically needs only left boxes
      - Uses sensor-data QoS to avoid backlog latency
      - Optional class filtering + confidence threshold
      - Visualization removed (keeps matching with downstream depth node)
    """

    def __init__(self):
        super().__init__("yolo_stereo_detector", namespace="follower_robot")
        self.bridge = CvBridge()

        # ---- Parameters ----
        self.declare_parameter("model_name", "yolo11n.pt")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf_th", 0.25)
        self.declare_parameter("max_hz", 10.0)  # 0 => no rate limit
        self.declare_parameter("use_left", True)
        self.declare_parameter("use_right", False)  # off by default (saves ~50% compute)
        self.declare_parameter("tracker_config", "botsort.yaml")  # or bytetrack.yaml
        self.declare_parameter("track", True)  # True => model.track, False => model.predict

        # Filter: set to empty list to allow all
        # Example: ["person", "car"]
        self.declare_parameter("allowed_classes", [])

        # Topics (keep your defaults to preserve matching)
        self.declare_parameter("left_topic", "/follower_robot/depth_cam/left/image_rect")
        self.declare_parameter("right_topic", "/follower_robot/depth_cam/right/image_rect")
        self.declare_parameter("left_out", "/follower_robot/depth_cam/left/detections")
        self.declare_parameter("right_out", "/follower_robot/depth_cam/right/detections")

        model_name = self.get_parameter("model_name").get_parameter_value().string_value
        imgsz = int(self.get_parameter("imgsz").get_parameter_value().integer_value)
        self.conf_th = float(self.get_parameter("conf_th").get_parameter_value().double_value)
        self.max_hz = float(self.get_parameter("max_hz").get_parameter_value().double_value)
        self.use_left = bool(self.get_parameter("use_left").get_parameter_value().bool_value)
        self.use_right = bool(self.get_parameter("use_right").get_parameter_value().bool_value)
        self.tracker_config = self.get_parameter("tracker_config").get_parameter_value().string_value
        self.do_track = bool(self.get_parameter("track").get_parameter_value().bool_value)

        allowed = self.get_parameter("allowed_classes").get_parameter_value().string_array_value
        self.allowed_classes = set(allowed) if allowed else None

        self.get_logger().info(f"Loading YOLO model: {model_name}")
        self.model = YOLO(model_name)

        # Ultralytics name map can be dict or list depending on version/model
        names = getattr(self.model.model, "names", None)
        if isinstance(names, dict):
            self.class_names = names
        elif isinstance(names, (list, tuple)):
            self.class_names = {i: n for i, n in enumerate(names)}
        else:
            self.class_names = {}

        self.imgsz = imgsz

        # ---- Subscriptions / pubs ----
        left_topic = self.get_parameter("left_topic").get_parameter_value().string_value
        right_topic = self.get_parameter("right_topic").get_parameter_value().string_value
        left_out = self.get_parameter("left_out").get_parameter_value().string_value
        right_out = self.get_parameter("right_out").get_parameter_value().string_value

        if self.use_left:
            self.create_subscription(Image, left_topic, lambda m: self._cb(m, "left"), qos_profile_sensor_data)

        if self.use_right:
            self.create_subscription(Image, right_topic, lambda m: self._cb(m, "right"), qos_profile_sensor_data)

        self.left_pub = self.create_publisher(Detection2DArray, left_out, 10)
        self.right_pub = self.create_publisher(Detection2DArray, right_out, 10)

        # Rate limiting per side
        self._last_run = {"left": 0.0, "right": 0.0}
        self._min_period = (1.0 / self.max_hz) if self.max_hz and self.max_hz > 0 else 0.0

        self.get_logger().info(
            f"YOLO detector ready (left={self.use_left}, right={self.use_right}, "
            f"track={self.do_track}, imgsz={self.imgsz}, conf_th={self.conf_th}, "
            f"max_hz={self.max_hz})"
        )

    def _cb(self, msg: Image, side: str):
        # Rate limit (keeps CPU stable; good when image is ~10 Hz)
        now = time.time()
        if self._min_period > 0.0 and (now - self._last_run[side]) < self._min_period:
            return
        self._last_run[side] = now

        # ROS -> OpenCV
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"[cv_bridge] failed on {side}: {e}")
            return

        if cv_img is None or not isinstance(cv_img, np.ndarray):
            self.get_logger().error(f"[{side}] invalid image from cv_bridge")
            return

        if cv_img.ndim != 3 or cv_img.shape[2] != 3:
            self.get_logger().error(f"[{side}] unexpected image shape {cv_img.shape}")
            return

        # Ensure contiguous uint8 (ultralytics likes contiguous arrays)
        if cv_img.dtype != np.uint8:
            cv_img = cv_img.astype(np.uint8)
        frame = np.ascontiguousarray(cv_img)

        # Inference
        try:
            t0 = time.time()
            if self.do_track:
                results = self.model.track(
                    source=frame,
                    persist=True,
                    tracker=self.tracker_config,
                    imgsz=self.imgsz,
                    conf=self.conf_th,
                    verbose=False,
                )
            else:
                results = self.model.predict(
                    source=frame,
                    imgsz=self.imgsz,
                    conf=self.conf_th,
                    verbose=False,
                )
            dt = time.time() - t0
            if PRINT:
                self.get_logger().info(f"YOLO {side}: {dt*1000.0:.1f} ms")
        except Exception as e:
            self.get_logger().error(f"[YOLO {side}] inference error: {e}")
            return

        out = Detection2DArray()
        out.header = msg.header

        if not results:
            self._publish(side, out)
            return

        res0 = results[0]
        boxes = getattr(res0, "boxes", None)
        if boxes is None or len(boxes) == 0:
            self._publish(side, out)
            return

        # Parse detections
        for box in boxes:
            # xyxy
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = [int(v) for v in xyxy.tolist()]
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            if w == 0 or h == 0:
                continue

            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = self.class_names.get(cls_id, str(cls_id))

            if self.allowed_classes is not None and cls_name not in self.allowed_classes:
                continue

            track_id: Optional[int] = None
            if getattr(box, "id", None) is not None:
                try:
                    track_id = int(box.id.cpu().numpy())
                except Exception:
                    track_id = None

            det = Detection2D()
            det.header = msg.header

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            det.bbox = BoundingBox2D()
            det.bbox.center.position.x = float(cx)
            det.bbox.center.position.y = float(cy)
            det.bbox.size_x = float(w)
            det.bbox.size_y = float(h)

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(cls_name)
            hyp.hypothesis.score = float(conf)

            # Important for your downstream: keep det.id as string
            det.id = str(track_id) if track_id is not None else ""
            det.results.append(hyp)

            out.detections.append(det)

        self._publish(side, out)

    def _publish(self, side: str, msg: Detection2DArray):
        if side == "left":
            self.left_pub.publish(msg)
        else:
            self.right_pub.publish(msg)


def main():
    rclpy.init()
    node = YoloStereoDetector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
