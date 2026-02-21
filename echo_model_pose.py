#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node

from gazebo_msgs.msg import ModelStates
from detectors_msgs.msg import GoalMarkerState


class EchoGoalWorldPoseWithGT(Node):
    def __init__(self):
        super().__init__("echo_goal_world_pose_with_gt")

        self.declare_parameter("robot_model_name", "my_robot")
        self.declare_parameter("goal_model_name", "goal_marker")
        self.declare_parameter("goal_state_topic", "/follower_robot/depth_cam/goal_marker_state")
        self.declare_parameter("print_hz", 5.0)

        # Camera offset wrt base_link (meters): x forward, y left
        self.declare_parameter("cam_x", 0.4)
        self.declare_parameter("cam_y", 0.0)

        self.robot_pose = None      # (x, y, yaw)
        self.goal_pose_gt = None    # (x, y)
        self.last_print_t = 0.0

        goal_topic = self.get_parameter("goal_state_topic").value
        self.sub_ms = self.create_subscription(ModelStates, "/model_states", self._model_states_cb, 10)
        self.sub_goal = self.create_subscription(GoalMarkerState, goal_topic, self._goal_state_cb, 10)

        self.get_logger().info(
            f"Listening: /model_states and {goal_topic}\n"
            f"robot_model_name='{self.get_parameter('robot_model_name').value}', "
            f"goal_model_name='{self.get_parameter('goal_model_name').value}'\n"
            f"cam_offset=(x={self.get_parameter('cam_x').value}, y={self.get_parameter('cam_y').value})\n"
            f"Assumption: GoalMarkerState.bearing_rad is camera bearing with RIGHT positive. "
            f"We convert to yaw-compatible sign by negating it."
        )

    def _model_states_cb(self, msg: ModelStates):
        robot_name = self.get_parameter("robot_model_name").value
        try:
            i = msg.name.index(robot_name)
            p = msg.pose[i].position
            o = msg.pose[i].orientation
            yaw = math.atan2(
                2.0 * (o.w * o.z + o.x * o.y),
                1.0 - 2.0 * (o.y * o.y + o.z * o.z),
            )
            self.robot_pose = (float(p.x), float(p.y), float(yaw))
        except ValueError:
            pass

        goal_name = self.get_parameter("goal_model_name").value
        try:
            j = msg.name.index(goal_name)
            gp = msg.pose[j].position
            self.goal_pose_gt = (float(gp.x), float(gp.y))
        except ValueError:
            pass

    def _goal_state_cb(self, st: GoalMarkerState):
        if not getattr(st, "visible", False):
            return
        if self.robot_pose is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        print_hz = float(self.get_parameter("print_hz").value)
        min_dt = 1.0 / max(0.1, print_hz)
        if (now - self.last_print_t) < min_dt:
            return
        self.last_print_t = now

        d = float(st.depth_m)
        if not math.isfinite(d) or d <= 0.0:
            return

        # Camera bearing (RIGHT positive). Convert to yaw-compatible sign (LEFT positive).
        b_cam = float(st.bearing_rad)
        b_yaw = -b_cam

        rx, ry, yaw = self.robot_pose

        cam_x = float(self.get_parameter("cam_x").value)
        cam_y = float(self.get_parameter("cam_y").value)

        # Camera origin in world
        rx_cam = rx + cam_x * math.cos(yaw) - cam_y * math.sin(yaw)
        ry_cam = ry + cam_x * math.sin(yaw) + cam_y * math.cos(yaw)

        # Project from camera with yaw-compatible bearing
        theta = yaw + b_yaw
        gx_est = rx_cam + d * math.cos(theta)
        gy_est = ry_cam + d * math.sin(theta)

        if self.goal_pose_gt is None:
            self.get_logger().info(
                f"[GOAL VISIBLE] est=(x={gx_est:.3f}, y={gy_est:.3f}) | "
                f"depth={d:.3f} | b_cam={b_cam:.3f} -> b_yaw={b_yaw:.3f} | "
                f"base=(x={rx:.3f}, y={ry:.3f}, yaw={yaw:.3f}) cam=(x={rx_cam:.3f}, y={ry_cam:.3f})"
            )
            return

        gx_gt, gy_gt = self.goal_pose_gt
        dx = gx_est - gx_gt
        dy = gy_est - gy_gt

        self.get_logger().info(
            f"[GOAL VISIBLE]\n"
            f"  est: (x={gx_est:.3f}, y={gy_est:.3f})\n"
            f"  gt : (x={gx_gt:.3f}, y={gy_gt:.3f})\n"
            f"  err: (dx={dx:.3f}, dy={dy:.3f})\n"
            f"  inputs: depth={d:.3f} m, b_cam={b_cam:.3f} rad -> b_yaw={b_yaw:.3f} rad | "
            f"base=(x={rx:.3f}, y={ry:.3f}, yaw={yaw:.3f}) cam=(x={rx_cam:.3f}, y={ry_cam:.3f})"
        )


def main():
    rclpy.init()
    node = EchoGoalWorldPoseWithGT()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
