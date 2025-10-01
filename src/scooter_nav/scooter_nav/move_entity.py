#!/usr/bin/env python3
"""
Move a Gazebo entity as a dynamic obstacle using /set_entity_state.

Params (declare via --ros-args -p name:=value):
  - entity_name (string): Gazebo entity to move (default: "moving_box")
  - frame        (string): Reference frame               (default: "world")
  - pattern      (string): "sine_y" | "sine_x" | "circle" (default: "sine_y")
  - x0, y0, z    (float):  Center/height                 (3.0, 2.0, 0.5)
  - A            (float):  Amplitude (meters)            (default: 2.0)
  - omega        (float):  Angular speed (rad/s)         (default: 0.4)
  - rate_hz      (float):  Command rate                  (default: 20.0)

Example:
  ros2 run scooter_nav move_entity --ros-args -p entity_name:=moving_box -p pattern:=circle -p A:=1.5 -p omega:=0.6
"""
import math
import time

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, Vector3


class MoveEntityNode(Node):
    def __init__(self):
        super().__init__('move_entity')

        # Parameters
        self.declare_parameter('entity_name', 'moving_box')
        self.declare_parameter('frame', 'world')
        self.declare_parameter('pattern', 'sine_y')  # sine_y | sine_x | circle
        self.declare_parameter('x0', 3.0)
        self.declare_parameter('y0', 2.0)
        self.declare_parameter('z', 0.5)
        self.declare_parameter('A', 2.0)
        self.declare_parameter('omega', 0.4)
        self.declare_parameter('rate_hz', 20.0)

        self.entity_name = self.get_parameter('entity_name').get_parameter_value().string_value
        self.frame = self.get_parameter('frame').get_parameter_value().string_value
        self.pattern = self.get_parameter('pattern').get_parameter_value().string_value
        self.x0 = self.get_parameter('x0').get_parameter_value().double_value
        self.y0 = self.get_parameter('y0').get_parameter_value().double_value
        self.z = self.get_parameter('z').get_parameter_value().double_value
        self.A = self.get_parameter('A').get_parameter_value().double_value
        self.omega = self.get_parameter('omega').get_parameter_value().double_value
        rate_hz = self.get_parameter('rate_hz').get_parameter_value().double_value

        # Service client
        self.cli = self.create_client(SetEntityState, '/set_entity_state')
        self.get_logger().info('Waiting for /set_entity_state service...')
        self.cli.wait_for_service()
        self.get_logger().info('Connected to /set_entity_state.')

        self.t0 = time.time()
        self.timer = self.create_timer(1.0 / max(rate_hz, 1.0), self._tick)

    def _tick(self):
        t = time.time() - self.t0
        x, y = self._plan(t)

        req = SetEntityState.Request()
        req.state = EntityState(
            name=self.entity_name,
            pose=Pose(
                position=Point(x=float(x), y=float(y), z=float(self.z)),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            ),
            twist=Twist(linear=Vector3(), angular=Vector3()),
            reference_frame=self.frame
        )
        self.cli.call_async(req)

    def _plan(self, t: float):
        if self.pattern == 'sine_x':
            return self.x0 + self.A * math.sin(self.omega * t), self.y0
        if self.pattern == 'circle':
            return self.x0 + self.A * math.cos(self.omega * t), self.y0 + self.A * math.sin(self.omega * t)
        # default: sine_y
        return self.x0, self.y0 + self.A * math.sin(self.omega * t)


def main():
    rclpy.init()
    node = MoveEntityNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
