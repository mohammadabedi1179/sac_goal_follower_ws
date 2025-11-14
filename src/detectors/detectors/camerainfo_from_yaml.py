#!/usr/bin/env python3
import rclpy, yaml, os
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import CameraInfo
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

class CIFromYAMLProxy(Node):
    def __init__(self):
        super().__init__('camerainfo_from_yaml', namespace="follower_robot")
        pkg = get_package_share_directory('detectors')
        left_path  = os.path.join(pkg, 'config', 'left.yaml')
        right_path = os.path.join(pkg, 'config', 'right.yaml')

        self.left_yaml  = self._load_yaml(left_path)
        self.right_yaml = self._load_yaml(right_path)

        # Latching so late subscribers get last value
        qos = QoSProfile(depth=1,
                         reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)

        # Subscribe to ORIGINAL infos from Gazebo
        self.create_subscription(CameraInfo, '/follower_robot/depth_cam/left/camera_info',  self._l_cb, 10)
        self.create_subscription(CameraInfo, '/follower_robot/depth_cam/right/camera_info', self._r_cb, 10)

        # Publish FIXED infos with SAME stamps
        self.pub_l = self.create_publisher(CameraInfo, '/follower_robot/depth_cam/left/camera_info_fixed',  qos)
        self.pub_r = self.create_publisher(CameraInfo, '/follower_robot/depth_cam/right/camera_info_fixed', qos)

        self.get_logger().info(f"Loaded:\n  {left_path}\n  {right_path}")
        self.get_logger().info(f"Right Tx (P[0,3]) = {self.right_yaml['p'][3]:.6f}")

    def _load_yaml(self, path):
        with open(path, 'r') as f:
            y = yaml.safe_load(f)
        return {
            'width':  int(y['image_width']),
            'height': int(y['image_height']),
            'model':  str(y.get('distortion_model', 'plumb_bob')),
            'k':      as_floats(y['camera_matrix']['data']),
            'd':      as_floats(y['distortion_coefficients']['data']),
            'r':      as_floats(y['rectification_matrix']['data']),
            'p':      as_floats(y['projection_matrix']['data']),
        }

    def _apply_and_pub(self, incoming: CameraInfo, yaml_data: dict, pub):
        out = CameraInfo()
        # copy timing & frame to keep sync with images
        out.header = incoming.header
        out.width  = yaml_data['width']
        out.height = yaml_data['height']
        out.distortion_model = yaml_data['model']
        out.k = list(yaml_data['k'])
        out.d = list(yaml_data['d'])
        out.r = list(yaml_data['r'])
        out.p = list(yaml_data['p'])
        pub.publish(out)

    def _l_cb(self, msg: CameraInfo):
        self._apply_and_pub(msg, self.left_yaml, self.pub_l)

    def _r_cb(self, msg: CameraInfo):
        self._apply_and_pub(msg, self.right_yaml, self.pub_r)

def as_floats(seq): 
    return [float(x) for x in seq]
def main():
    rclpy.init()
    node = CIFromYAMLProxy()
    # Respect sim time if provided in launch
    if not node.has_parameter('use_sim_time'): node.declare_parameter('use_sim_time', False)
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()