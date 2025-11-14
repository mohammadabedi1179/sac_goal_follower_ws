#!/usr/bin/env python3
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
from cv_bridge import CvBridge, CvBridgeError

class RedMarkerDetector(Node):
    def __init__(self):
        super().__init__('red_marker_detector')

        # ---- Parameters ----
        self.declare_parameter('input_topic', '/depth_cam/left/image_raw')
        self.declare_parameter('show_window', True)
        self.declare_parameter('min_area', 500)  # pixels
        # HSV thresholds for red (two ranges because hue wraps)
        self.declare_parameter('lower_red_1', [0, 100, 100])
        self.declare_parameter('upper_red_1', [10, 255, 255])
        self.declare_parameter('lower_red_2', [170, 100, 100])
        self.declare_parameter('upper_red_2', [180, 255, 255])
        self.declare_parameter('morph_kernel', 5)

        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, input_topic, self.image_cb, qos)
        self.roi_pub = self.create_publisher(Int32MultiArray, 'goal_marker/roi', 10)

        self.get_logger().info(f'Listening for images on: {input_topic}')
        self.get_logger().info('Press Q in the OpenCV window to quit.')

    def _params(self):
        p = self.get_parameter
        # Allow runtime tuning via `ros2 param set`
        lr1 = np.array(p('lower_red_1').value, dtype=np.uint8)
        ur1 = np.array(p('upper_red_1').value, dtype=np.uint8)
        lr2 = np.array(p('lower_red_2').value, dtype=np.uint8)
        ur2 = np.array(p('upper_red_2').value, dtype=np.uint8)
        min_area = int(p('min_area').value)
        morph_kernel = int(p('morph_kernel').value)
        show_window = bool(p('show_window').value)
        return lr1, ur1, lr2, ur2, min_area, morph_kernel, show_window

    def image_cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge error: {e}')
            return

        lr1, ur1, lr2, ur2, min_area, morph_kernel, show_window = self._params()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, lr1, ur1)
        mask2 = cv2.inRange(hsv, lr2, ur2)
        mask = cv2.bitwise_or(mask1, mask2)

        if morph_kernel > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                (cx, cy), radius = cv2.minEnclosingCircle(c)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (int(cx), int(cy)), int(radius), (255, 0, 0), 2)
                cv2.putText(frame, 'Goal Marker', (x, max(0, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                self.roi_pub.publish(Int32MultiArray(data=[int(x), int(y), int(w), int(h)]))

        if show_window:
            vis = frame.copy()
            try:
                mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                h, w = mask_color.shape[:2]
                inset_h, inset_w = min(180, h), min(240, w)
                vis[0:inset_h, 0:inset_w] = cv2.resize(mask_color, (inset_w, inset_h))
                cv2.rectangle(vis, (0, 0), (inset_w, inset_h), (200, 200, 200), 1)
                cv2.putText(vis, 'mask', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            except Exception:
                pass

            cv2.imshow('Goal Marker Detection (left cam)', vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info('Shutting down on user request (Q).')
                rclpy.shutdown()
                cv2.destroyAllWindows()

def main():
    rclpy.init()
    node = RedMarkerDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down (KeyboardInterrupt).')
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
