#!/usr/bin/env python3
import math
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

MAX_LINEAR_ACCEL = 0.2    # m/s²
MAX_ANGULAR_ACCEL = 0.5   # rad/s²

def yaw_from_quat(q):
    # q: geometry_msgs/Quaternion
    s = 2.0*(q.w*q.z + q.x*q.y)
    c = 1.0 - 2.0*(q.y*q.y + q.z*q.z)
    return math.atan2(s, c)

def wrap(a):
    while a > math.pi:  a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

class GoToZoneAndSearch(Node):
    def __init__(self):
        super().__init__('go_to_zone_and_search')

        # --- Params (override with --ros-args -p name:=value) ---
        self.declare_parameter('goal_x', 2.0)
        self.declare_parameter('goal_y', 1.0)
        self.declare_parameter('zone_radius', 0.7)
        self.declare_parameter('v_max', 0.5)
        self.declare_parameter('w_max', 1.5)
        self.declare_parameter('k_v', 0.7)
        self.declare_parameter('k_w', 2.0)
        self.declare_parameter('search_spin_speed', 0.5)
        self.declare_parameter('align_gain', 1.2)
        self.declare_parameter('approach_distance', 0.45)   # stop this far from marker
        # HSV for green marker (tune if needed)
        self.declare_parameter('h_low', 35)
        self.declare_parameter('h_high', 85)
        self.declare_parameter('s_low', 80)
        self.declare_parameter('v_low', 80)
        self.declare_parameter('min_area', 300)             # px area to accept a blob

        self.goal_x = float(self.get_parameter('goal_x').value)
        self.goal_y = float(self.get_parameter('goal_y').value)
        self.zone_radius = float(self.get_parameter('zone_radius').value)
        self.v_max = float(self.get_parameter('v_max').value)
        self.w_max = float(self.get_parameter('w_max').value)
        self.k_v = float(self.get_parameter('k_v').value)
        self.k_w = float(self.get_parameter('k_w').value)
        self.search_spin_speed = float(self.get_parameter('search_spin_speed').value)
        self.align_gain = float(self.get_parameter('align_gain').value)
        self.approach_distance = float(self.get_parameter('approach_distance').value)
        self.h_low = int(self.get_parameter('h_low').value)
        self.h_high = int(self.get_parameter('h_high').value)
        self.s_low = int(self.get_parameter('s_low').value)
        self.v_low = int(self.get_parameter('v_low').value)
        self.min_area = int(self.get_parameter('min_area').value)

        # --- I/O ---
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.rgb_sub  = self.create_subscription(Image, '/depth_cam/image_raw', self.rgb_cb, qos_profile_sensor_data)
        self.depth_sub= self.create_subscription(Image, '/depth_cam/depth/image_raw', self.depth_cb, qos_profile_sensor_data)

        self.bridge = CvBridge()

        # --- State ---
        self.state = 'NAVIGATE_TO_ZONE'  # or SEARCH_MARKER or DONE
        self.x = self.y = 0.0
        self.yaw = 0.0
        self.last_odom_time = time.time()
        self.rgb = None
        self.depth = None
        self.depth_encoding = None
        self.marker_seen = False
        self.marker_cxy = (None, None)
        self.marker_dist = None
        self.img_shape = (0, 0)

        self.timer = self.create_timer(0.05, self.loop)  # 20 Hz

        self.get_logger().info(f"Goal zone center=({self.goal_x:.2f},{self.goal_y:.2f}), R={self.zone_radius:.2f}")

        self.current_linear = 0.0
        self.current_angular = 0.0
        self.last_time = self.get_clock().now()


    # --- Callbacks ---
    def odom_cb(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = yaw_from_quat(msg.pose.pose.orientation)
        self.last_odom_time = time.time()

    def rgb_cb(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.img_shape = cv_img.shape[:2]
            self.marker_seen, self.marker_cxy = self.detect_green(cv_img)
            self.rgb = cv_img
        except Exception as e:
            self.get_logger().warn(f"RGB convert error: {e}")

    def depth_cb(self, msg: Image):
        self.depth_encoding = msg.encoding
        try:
            if msg.encoding == '32FC1':
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            elif msg.encoding == '16UC1':
                d16 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
                depth = d16.astype(np.float32) / 1000.0  # mm->m
            else:
                # Try a generic conversion
                depth = self.bridge.imgmsg_to_cv2(msg).astype(np.float32)
            self.depth = depth
        except Exception as e:
            self.get_logger().warn(f"Depth convert error: {e}")

    # --- Vision: simple green blob detection ---
    def detect_green(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([self.h_low, self.s_low, self.v_low], dtype=np.uint8)
        upper = np.array([self.h_high, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.medianBlur(mask, 5)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return False, (None, None)
        # largest blob
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < self.min_area:
            return False, (None, None)
        M = cv2.moments(c)
        if M['m00'] == 0:
            return False, (None, None)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        return True, (cx, cy)

    # --- Main loop ---
    def loop(self):
        # failsafe: if odom stale, stop
        if time.time() - self.last_odom_time > 1.0:
            self.stop()
            return

        if self.state == 'NAVIGATE_TO_ZONE':
            self.navigate_to_zone()
        elif self.state == 'SEARCH_MARKER':
            self.search_and_approach_marker()
        elif self.state == 'DONE':
            self.stop()
        else:
            self.stop()

    def navigate_to_zone(self):
        dx = self.goal_x - self.x
        dy = self.goal_y - self.y
        dist = math.hypot(dx, dy)
        self.get_logger().debug(f"Navigating to zone: dx={dx:.2f}, dy={dy:.2f}, dist={dist:.2f}")

        if dist <= self.zone_radius:
            self.get_logger().info("Entered goal zone. Switching to SEARCH_MARKER.")
            self.state = 'SEARCH_MARKER'
            self.stop()
            return

        target_yaw = math.atan2(dy, dx)
        yaw_err = wrap(target_yaw - self.yaw)
        self.get_logger().debug(f"Target yaw={target_yaw:.2f}, Current yaw={self.yaw:.2f}, Yaw error={yaw_err:.2f}")

        v = max(0.0, min(self.k_v * dist, self.v_max))
        if abs(yaw_err) > math.pi/4:
            v *= 0.2

        w = max(-self.w_max, min(self.k_w * yaw_err, self.w_max))
        self.get_logger().debug(f"Velocity command: v={v:.2f}, w={w:.2f}")

        self.publish_cmd(v, w)

    def search_and_approach_marker(self):
        if not self.marker_seen:
            self.get_logger().debug("Marker not seen. Rotating in place.")
            self.publish_cmd(0.0, self.search_spin_speed)
            return

        h, w_img = self.img_shape
        cx, cy = self.marker_cxy
        self.get_logger().debug(f"Marker seen at: cx={cx}, cy={cy}")

        if cx is None or self.depth is None:
            self.get_logger().debug("Marker position or depth data unavailable. Rotating in place.")
            self.publish_cmd(0.0, self.search_spin_speed)
            return

        cx_norm = (cx - w_img/2) / (w_img/2)
        w_cmd = max(-self.w_max, min(self.align_gain * cx_norm, self.w_max))

        d = float(self.depth[cy, cx]) if 0 <= cy < self.depth.shape[0] and 0 <= cx < self.depth.shape[1] else float('nan')
        self.get_logger().debug(f"Marker distance: d={d:.2f}")

        if not (0.05 < d < 20.0) or math.isnan(d):
            self.get_logger().debug("Invalid marker distance. Aligning only.")
            self.publish_cmd(0.0, w_cmd)
            return

        self.marker_dist = d

        if d <= self.approach_distance:
            self.get_logger().info("Reached marker. DONE.")
            self.state = 'DONE'
            self.stop()
            return

        v_cmd = min(self.v_max * 0.5, max(0.05, 0.6 * (d - self.approach_distance)))
        if abs(cx_norm) > 0.2:
            v_cmd *= 0.5

        self.get_logger().debug(f"Velocity command: v_cmd={v_cmd:.2f}, w_cmd={w_cmd:.2f}")
        self.publish_cmd(v_cmd, w_cmd)

    def publish_cmd(self, v, w):
        # compute time step
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now

        # apply ramp filtering
        self.current_linear = self.ramped_velocity(v, self.current_linear, MAX_LINEAR_ACCEL, dt)
        self.current_angular = self.ramped_velocity(w, self.current_angular, MAX_ANGULAR_ACCEL, dt)

        # publish smoothed cmd
        msg = Twist()
        msg.linear.x = float(self.current_linear)
        msg.angular.z = float(self.current_angular)
        self.cmd_pub.publish(msg)

    def stop(self):
        self.publish_cmd(0.0, 0.0)

    def ramped_velocity(self, target, current, max_accel, dt):
        step = max_accel * dt
        if abs(target - current) < step:
            return target
        return current + step if target > current else current - step

def main():
    rclpy.init()
    node = GoToZoneAndSearch()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.stop()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()