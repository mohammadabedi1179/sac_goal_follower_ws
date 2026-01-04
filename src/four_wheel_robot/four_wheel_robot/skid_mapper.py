#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray


class SkidMapper(Node):
    def __init__(self):
        super().__init__("skid_mapper")

        # Set to YOUR real geometry
        self.track_width = 0.66108        # meters
        self.wheel_radius = 0.185 / 2.0   # meters

        # Input
        self.sub = self.create_subscription(
            Twist, "/follower_robot/cmd_vel", self.on_cmd, 10
        )

        # Output to JointGroupVelocityController
        self.pub = self.create_publisher(
            Float64MultiArray,
            "/follower_robot/wheels_velocity_controller/commands",
            10
        )

        # Stop if cmd_vel disappears
        self.timeout_s = 0.4
        self.last_time = self.get_clock().now()
        self.last_cmd = Twist()
        self.timer = self.create_timer(0.02, self.on_timer)  # 50 Hz

        self.get_logger().info("Skid mapper is running (cmd_vel -> 4 wheel velocities).")

    def on_cmd(self, msg: Twist):
        self.last_cmd = msg
        self.last_time = self.get_clock().now()
        self.publish_wheels(msg)

    def on_timer(self):
        age = (self.get_clock().now() - self.last_time).nanoseconds * 1e-9
        if age > self.timeout_s:
            self.publish_wheels(Twist())

    def publish_wheels(self, cmd: Twist):
        v = float(cmd.linear.x)
        wz = float(cmd.angular.z)

        w_left  = (v - wz * self.track_width / 2.0) / self.wheel_radius
        w_right = (v + wz * self.track_width / 2.0) / self.wheel_radius

        # Order MUST match controllers.yaml joints: [FL, BL, FR, BR]
        msg = Float64MultiArray()
        msg.data = [w_left, w_left, w_right, w_right]
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = SkidMapper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
