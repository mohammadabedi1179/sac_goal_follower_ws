#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Range
from std_msgs.msg import Float32
import math
import time

class UltrasonicObstacleDistance(Node):
    def __init__(self):
        super().__init__("ultrasonic_obstacle_distance", namespace="follower_robot")

        base = "/follower_robot/ultrasonic_bridge"
        self.sensors = {
            "front_left":  f"{base}/front_left/range",
            "front_right": f"{base}/front_right/range",
            "left_side":   f"{base}/left_side/range",
            "right_side":  f"{base}/right_side/range",
        }

        self.last = {}  # name -> (t, range_m, min_m, max_m)

        for name, topic in self.sensors.items():
            self.get_logger().info(f"Subscribing to {topic} as '{name}'")
            self.create_subscription(
                Range, topic,
                lambda msg, n=name: self._cb_range(n, msg),
                qos_profile_sensor_data
            )

        # publish “closest obstacle” distance (optional)
        self.closest_pub = self.create_publisher(Float32, f"{base}/closest_distance_m", 10)
        self.timer = self.create_timer(0.1, self._publish_closest)

        self.get_logger().info("UltrasonicObstacleDistance node READY")

    def _cb_range(self, name: str, msg: Range):
        # Range.msg fields: radiation_type, field_of_view, min_range, max_range, range
        r = float(msg.range)

        # Some sensors output +inf / NaN when nothing is detected
        if not math.isfinite(r):
            r = msg.max_range

        self.last[name] = (time.time(), r, msg.min_range, msg.max_range)

        # Log only when “something is near” to avoid spam
        if r < (msg.max_range - 1e-3):
            self.get_logger().info(f"[{name}] obstacle at {r:.3f} m (max={msg.max_range:.2f})")

    def _publish_closest(self):
        if not self.last:
            return

        # ignore stale data (>1s old)
        now = time.time()
        vals = [r for (t, r, _, _) in self.last.values() if (now - t) < 1.0]
        if not vals:
            return

        closest = min(vals)
        out = Float32()
        out.data = float(closest)
        self.closest_pub.publish(out)

def main():
    rclpy.init()
    node = UltrasonicObstacleDistance()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
