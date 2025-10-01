#!/usr/bin/env python3
"""
Minimal obstacle avoidance using a PointCloud2 (e.g., from stereo_image_proc /points2).

It watches a forward window and:
  - drives forward at fwd_speed when clear
  - turns in place at turn_speed when something is closer than stop_dist

Params:
  - cloud_topic (string): PointCloud2 topic (default: "/points2")
  - cmd_vel_topic (string): Twist output (default: "/cmd_vel")
  - stop_dist (float): meters (default: 0.8)
  - fwd_speed (float): m/s (default: 0.4)
  - turn_speed (float): rad/s (default: 0.6)
  - window_y (float): |y| < window_y in meters (default: 0.4)
  - window_z_min (float): min z (default: 0.0)
  - window_z_max (float): max z (default: 0.5)
  - sample_stride (int): downsample points (>=1) (default: 4)

Run:
  ros2 run scooter_nav min_avoid
"""
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2

# Use the helper to iterate PointCloud2 safely
try:
    from sensor_msgs_py import point_cloud2
except Exception:  # fallback name on some installs
    from sensor_msgs import point_cloud2  # type: ignore


class MinAvoidNode(Node):
    def __init__(self):
        super().__init__('min_avoid')

        # Parameters
        self.declare_parameter('cloud_topic', '/points2')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('stop_dist', 0.8)
        self.declare_parameter('fwd_speed', 0.4)
        self.declare_parameter('turn_speed', 0.6)
        self.declare_parameter('window_y', 0.4)
        self.declare_parameter('window_z_min', 0.0)
        self.declare_parameter('window_z_max', 0.5)
        self.declare_parameter('sample_stride', 4)

        cloud_topic = self.get_parameter('cloud_topic').get_parameter_value().string_value
        cmd_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value
        self.stop_dist = self.get_parameter('stop_dist').get_parameter_value().double_value
        self.fwd_speed = self.get_parameter('fwd_speed').get_parameter_value().double_value
        self.turn_speed = self.get_parameter('turn_speed').get_parameter_value().double_value
        self.window_y = self.get_parameter('window_y').get_parameter_value().double_value
        self.win_z_min = self.get_parameter('window_z_min').get_parameter_value().double_value
        self.win_z_max = self.get_parameter('window_z_max').get_parameter_value().double_value
        self.sample_stride = max(1, int(self.get_parameter('sample_stride').get_parameter_value().integer_value or 4))

        self.pub = self.create_publisher(Twist, cmd_topic, 10)
        self.sub = self.create_subscription(PointCloud2, cloud_topic, self._cloud_cb, 10)

        self.get_logger().info(f"Listening to {cloud_topic}, publishing {cmd_topic}")

    def _cloud_cb(self, msg: PointCloud2):
        # Iterate XYZ, skipping NaNs; downsample with sample_stride for speed
        min_x = math.inf
        for idx, pt in enumerate(point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)):
            if (idx % self.sample_stride) != 0:
                continue
            x, y, z = pt
            if x <= 0.0:
                continue
            if abs(y) > self.window_y:
                continue
            if not (self.win_z_min < z < self.win_z_max):
                continue
            if x < min_x:
                min_x = x

        cmd = Twist()
        if min_x < self.stop_dist:
            cmd.linear.x = 0.0
            cmd.angular.z = self.turn_speed
        else:
            cmd.linear.x = self.fwd_speed
            cmd.angular.z = 0.0
        self.pub.publish(cmd)


def main():
    rclpy.init()
    node = MinAvoidNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
