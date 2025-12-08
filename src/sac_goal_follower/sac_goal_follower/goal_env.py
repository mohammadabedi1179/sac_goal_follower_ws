import time
import math
import json
import subprocess

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

from gymnasium import Env, spaces
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String

from detectors_msgs.msg import GoalMarkerState


def _wrap(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


class _ROS(Node):
    """
    Thin ROS2 wrapper for the SAC environment.

    - Subscribes to GoalMarkerState (vision + depth results).
    - Subscribes to robot & goal odometry.
    - Subscribes to obstacle depth JSON.
    - Publishes cmd_vel.
    """

    def __init__(
        self,
        cmd_topic: str,
        goal_state_topic: str,
        goal_odom_topic: str,
        robot_odom_topic: str,
        obstacle_topic: str = "/follower_robot/obstacles_depth",
    ):
        super().__init__("sac_goal_env_node")

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)

        # Buffers
        self.goal_state: GoalMarkerState | None = None
        self.goal_pose = None                 # (x, y) from /goal_marker/odom
        self.robot_pose = None                # (x, y, yaw) from /follower_robot/odom
        self._last_odom_vx = 0.0
        self._last_odom_wz = 0.0
        self.obstacles_json: str | None = None

        # Subscriptions
        self.create_subscription(
            GoalMarkerState,
            goal_state_topic,
            self._goal_state_cb,
            10,
        )
        self.create_subscription(Odometry, goal_odom_topic, self._goal_odom_cb, 10)
        self.create_subscription(Odometry, robot_odom_topic, self._robot_odom_cb, 10)
        self.create_subscription(String, obstacle_topic, self._obstacle_cb, 10)

    # --- Callbacks ---

    def _goal_state_cb(self, msg: GoalMarkerState):
        self.goal_state = msg

    def _goal_odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self.goal_pose = (p.x, p.y)

    def _robot_odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (o.w * o.z + o.x * o.y),
            1.0 - 2.0 * (o.y * o.y + o.z * o.z),
        )
        self.robot_pose = (p.x, p.y, yaw)
        self._last_odom_vx = msg.twist.twist.linear.x
        self._last_odom_wz = msg.twist.twist.angular.z

    def _obstacle_cb(self, msg: String):
        # Raw JSON string from stereo_box_depth_from_disparity_IQR_EMA
        self.obstacles_json = msg.data

    # --- Helpers ---

    def send_cmd(self, v: float, w: float) -> None:
        tw = Twist()
        tw.linear.x = float(v)
        tw.angular.z = float(w)
        self.cmd_pub.publish(tw)


class GoalFollowerEnv(Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        cmd_topic="/follower_robot/cmd_vel",
        goal_state_topic="/follower_robot/depth_cam/goal_marker_state",
        goal_odom_topic="/goal_marker/odom",
        robot_odom_topic="/follower_robot/odom",
        wheel_radius=0.0925,
        wheel_separation=0.66108,
        dt=0.1,
        lost_timeout=5.0,
        success_radius=0.35,
        time_limit=20.0,
        c_time=0.01,
        c_dist=0.1,
        c_lost=0.1,
        R_goal=50.0,
    ):
        super().__init__()

        # --------- ACTION SPACE ----------
        self.v_max = 2.0     # [m/s]
        self.w_max = 2.0     # [rad/s]
        self.action_low = np.array([-1.0, -1.0], dtype=np.float32)
        self.action_high = np.array([1.0, 1.0], dtype=np.float32)
        self.action_space = spaces.Box(
            low=self.action_low,
            high=self.action_high,
            dtype=np.float32,
        )

        # --------- OBSERVATION SPACE ----------
        # [distance_to_goal, bearing_to_goal, min_obstacle_depth]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -math.pi, 0.0], np.float32),
            high=np.array([np.inf, math.pi, np.inf], np.float32),
        )

        # Kinematics
        self.r = wheel_radius
        self.L = wheel_separation

        self.dt = dt
        self.lost_timeout = lost_timeout
        self.success_radius = success_radius
        self.time_limit = time_limit

        # Reward parameters
        self.c_time = c_time
        self.c_dist = c_dist
        self.c_lost = c_lost
        self.R_goal = R_goal
        self.c_angle = 0.05     # penalty on |bearing|
        self.c_progress = 0.5   # reward for reducing distance
        self.c_ctrl = 0.01      # control effort penalty

        # Action smoothing
        self.smooth_alpha = 0.6
        self._prev_v_cmd = 0.0
        self._prev_w_cmd = 0.0

        # Search-on-reset
        self.enable_search_on_reset = True
        self.search_angular_speed = 4.0  # [rad/s]

        # ROS wrapper
        self.ros = _ROS(
            cmd_topic=cmd_topic,
            goal_state_topic=goal_state_topic,
            goal_odom_topic=goal_odom_topic,
            robot_odom_topic=robot_odom_topic,
            obstacle_topic="/follower_robot/obstacles_depth",
        )
        self.exec = SingleThreadedExecutor()
        self.exec.add_node(self.ros)

        self._t0 = None
        self._last_seen = None
        self._last_goal_pose = None
        self._visible = False

        # For progress-based reward
        self._prev_dist = None

        # Per-episode logging
        self._ep_robot_traj = []
        self._robot_start = None
        self._goal_start = None
        self._robot_last = None
        self._goal_last = None
        self._min_dist = None

        # Obstacle handling
        self.obstacle_safe_radius = 1.5       # start penalizing here [m]
        self.obstacle_collision_radius = 0.7  # collision radius [m]
        self.c_obstacle = 2.0                 # penalty factor
        self.max_obstacle_depth = 10.0        # default "no obstacle" depth
        self._last_obs_valid = np.array(
            [7.08, 0.0, self.max_obstacle_depth], np.float32
        )

    # --- ROS spin helper ---

    def _spin(self, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            self.exec.spin_once(timeout_sec=0.001)

    # --- Obstacle depth helper ---

    def _compute_min_obstacle_depth(self) -> float:
        """
        Parse latest JSON from obstacles_depth and return min depth [m].
        JSON format: [{"id": "...", "class": "...", "disparity_px": ..., "depth_m": ...}, ...]
        """
        if getattr(self.ros, "obstacles_json", None) is None:
            return self.max_obstacle_depth

        try:
            objs = json.loads(self.ros.obstacles_json)
        except Exception:
            return self.max_obstacle_depth

        if not isinstance(objs, list) or len(objs) == 0:
            return self.max_obstacle_depth

        depths = []
        for o in objs:
            try:
                d = float(o.get("depth_m", float("inf")))
                if math.isfinite(d) and d > 0.0:
                    depths.append(d)
            except Exception:
                continue

        if not depths:
            return self.max_obstacle_depth

        d_min = min(depths)
        d_min = max(0.0, min(d_min, self.max_obstacle_depth))
        return float(d_min)

    # --- Observation construction ---

    def _obs(self, default: bool = False):
        """
        Build observation [distance_to_goal, bearing, min_obstacle_depth].

        If default=True, return last valid observation without forcing new perception.
        """
        if default:
            return self._last_obs_valid.copy()

        st = self.ros.goal_state
        self._visible = False

        # CASE 1: directly visible from GoalMarkerState
        if st is not None and st.visible:
            self._visible = True
            self._last_seen = time.time()

            d = float(st.depth_m)
            b = _wrap(float(st.bearing_rad))

            # Estimate world-frame goal pose from robot pose + (d, b)
            if self.ros.robot_pose is not None:
                rx, ry, ryaw = self.ros.robot_pose
                gx = rx + d * math.cos(ryaw + b)
                gy = ry + d * math.sin(ryaw + b)
                self._last_goal_pose = (gx, gy)

            min_obs_depth = self._compute_min_obstacle_depth()
            obs = np.array([d, b, min_obs_depth], np.float32)
            self._last_obs_valid = obs
            return obs

        # CASE 2: not visible, but we have last known goal pose in world frame
        if self._last_goal_pose is not None and self.ros.robot_pose is not None:
            rx, ry, ryaw = self.ros.robot_pose
            dx = self._last_goal_pose[0] - rx
            dy = self._last_goal_pose[1] - ry
            dist = math.sqrt(dx * dx + dy * dy)
            bearing = _wrap(math.atan2(dy, dx) - ryaw)
            min_obs_depth = self._compute_min_obstacle_depth()
            obs = np.array([dist, bearing, min_obs_depth], np.float32)
            self._last_obs_valid = obs
            return obs

        # CASE 3: fallback
        return self._last_obs_valid.copy()

    # --- Gazebo reset helper ---

    def _reset_entity_with_retry(
        self,
        name: str,
        x: float,
        y: float,
        z: float = 0.0,
        yaw: float = 0.0,
        max_attempts: int = 3,
    ) -> bool:
        for attempt in range(max_attempts):
            self.ros.get_logger().info(
                f"[{name}] Reset attempt {attempt + 1}/{max_attempts}"
            )
            qz, qw = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
            cmd = [
                "ros2",
                "service",
                "call",
                "/set_entity_state",
                "gazebo_msgs/srv/SetEntityState",
                (
                    "{state: {name: '"
                    + name
                    + "', pose: {position: {x: "
                    + str(x)
                    + ", y: "
                    + str(y)
                    + ", z: "
                    + str(z)
                    + "}, orientation: {z: "
                    + str(qz)
                    + ", w: "
                    + str(qw)
                    + "}}}}"
                ),
            ]
            try:
                out = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=5
                )
                if "success: True" in out.stdout or "success=True" in out.stdout:
                    self.ros.get_logger().info(
                        f"[{name}] Reset confirmed by CLI"
                    )
                    return True
            except subprocess.TimeoutExpired:
                self.ros.get_logger().warn(
                    f"[{name}] CLI call timed out"
                )
        self.ros.get_logger().error(
            f"[{name}] Failed to reset after {max_attempts}"
        )
        return False

    def _search_full_rotation(self) -> bool:
        """
        Rotate robot 360° at fixed angular speed, stop if goal becomes visible.
        """
        if self.ros.robot_pose is None:
            self.ros.get_logger().warn(
                "[SEARCH] No robot odom available, skipping search."
            )
            return False

        _, _, yaw0 = self.ros.robot_pose
        target_rotation = 2.0 * math.pi
        w = self.search_angular_speed

        self.ros.get_logger().info(
            f"[SEARCH] Starting 360-degree search at {w:.2f} rad/s"
        )

        accumulated = 0.0
        prev_yaw = yaw0
        seen_once = False

        while accumulated < target_rotation:
            self.ros.send_cmd(0.0, w)
            self._spin(self.dt)

            # Check visibility via perception
            _ = self._obs(default=False)
            if self._visible:
                self.ros.get_logger().info("[SEARCH] Goal seen during search.")
                seen_once = True
                break

            if self.ros.robot_pose is not None:
                _, _, yaw = self.ros.robot_pose
                dyaw = _wrap(yaw - prev_yaw)
                accumulated += abs(dyaw)
                prev_yaw = yaw

        self.ros.send_cmd(0.0, 0.0)
        self._spin(0.1)

        self.ros.get_logger().info(
            f"[SEARCH] Finished search; seen={seen_once}, total_rot={accumulated:.2f} rad"
        )
        return seen_once

    # --- Gym API: reset ---

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._t0 = None
        self._last_seen = None
        self._visible = False
        self._last_goal_pose = None

        self._ep_robot_traj = []
        self._robot_start = None
        self._goal_start = None
        self._robot_last = None
        self._goal_last = None
        self._min_dist = None

        self._prev_v_cmd = 0.0
        self._prev_w_cmd = 0.0
        self._prev_dist = None

        self.ros.send_cmd(0.0, 0.0)

        # Reset robot and goal in Gazebo
        self._reset_entity_with_retry("my_robot", 0.0, 0.0, 0.3, 0.7854)

        # Reset random obstacles via your service
        subprocess.run(
            [
                "ros2",
                "service",
                "call",
                "/reset_random_obstacles",
                "std_srvs/srv/Trigger",
                "{}",
            ],
            capture_output=True,
            text=True,
        )
        self.ros.get_logger().info("[ENV] Random obstacles reset")

        # Random goal on circle of radius 7 m
        radius = 7.0
        angle = self.np_random.uniform(0.0, 2.0 * math.pi)
        gx = radius * math.cos(angle)
        gy = radius * math.sin(angle)
        self._reset_entity_with_retry("goal_marker", gx, gy, 0.75, 0.0)

        self._spin(0.5)

        # Search for goal
        seen = False
        if self.enable_search_on_reset:
            seen = self._search_full_rotation()

        now = time.time()
        self._t0 = now
        self._last_seen = now

        if self.ros.robot_pose is not None:
            rx, ry, _ = self.ros.robot_pose
            self._robot_start = (rx, ry)
            self._robot_last = (rx, ry)
            self._ep_robot_traj.append((rx, ry))

        if self.ros.goal_pose is not None:
            gx0, gy0 = self.ros.goal_pose
            self._goal_start = (gx0, gy0)
            self._goal_last = (gx0, gy0)

        obs = self._obs(default=not seen)
        self._prev_dist = float(obs[0])

        self.ros.get_logger().info(
            f"Episode reset completed: initial obs={obs}, "
            f"goal_spawn=({gx:.2f}, {gy:.2f}), seen_in_search={seen}"
        )
        return obs, {}

    # --- Gym API: step ---

    def step(self, action):
        # Clip & map action
        a = np.clip(action, self.action_low, self.action_high).astype(np.float32)
        v_des = float(a[0]) * self.v_max
        w_des = float(a[1]) * self.w_max

        # Smooth commands
        v_cmd = (1.0 - self.smooth_alpha) * self._prev_v_cmd + self.smooth_alpha * v_des
        w_cmd = (1.0 - self.smooth_alpha) * self._prev_w_cmd + self.smooth_alpha * w_des
        self._prev_v_cmd = v_cmd
        self._prev_w_cmd = w_cmd

        self.ros.send_cmd(v_cmd, w_cmd)
        self._spin(self.dt)

        obs = self._obs()
        dist, bearing, min_obs_depth = float(obs[0]), float(obs[1]), float(obs[2])

        # Ground-truth distance (for logging)
        real_dist = float("inf")
        if self.ros.robot_pose is not None and self.ros.goal_pose is not None:
            rx, ry, _ = self.ros.robot_pose
            gx, gy = self.ros.goal_pose
            real_dist = math.sqrt((gx - rx) ** 2 + (gy - ry) ** 2)

        vx = self.ros._last_odom_vx
        wz = self.ros._last_odom_wz

        if self.ros.robot_pose is not None:
            rx, ry, _ = self.ros.robot_pose
            self._robot_last = (rx, ry)
            self._ep_robot_traj.append((rx, ry))
        if self.ros.goal_pose is not None:
            gx, gy = self.ros.goal_pose
            self._goal_last = (gx, gy)

        if self._min_dist is None or dist < self._min_dist:
            self._min_dist = dist

        # --------- REWARD ----------
        prev_dist = self._prev_dist if self._prev_dist is not None else dist
        progress = prev_dist - dist
        self._prev_dist = dist

        reward = 0.0
        reward -= self.c_time * self.dt

        if self._visible:
            reward += self.c_progress * progress
            reward -= self.c_dist * dist
            reward -= self.c_angle * abs(bearing)
        else:
            reward -= self.c_lost * self.dt

        reward -= self.c_ctrl * (abs(v_cmd) + abs(w_cmd))

        term = False
        trunc = False
        reason = ""

        # Obstacle penalties / collision
        if min_obs_depth < self.obstacle_collision_radius:
            reward -= self.R_goal
            term = True
            reason = "Collision with obstacle"
        elif min_obs_depth < self.obstacle_safe_radius:
            reward -= self.c_obstacle * (self.obstacle_safe_radius - min_obs_depth)

        # Success
        if not term and self._visible and dist <= self.success_radius:
            reward += self.R_goal
            term = True
            reason = "Reached goal"

        # Lost marker too long
        if (time.time() - self._last_seen) >= self.lost_timeout:
            reward -= self.R_goal
            term = True
            reason = "Lost marker timeout"

        # Time limit
        if (time.time() - self._t0) >= self.time_limit:
            trunc = True
            reason = "Time limit reached"

        self.ros.get_logger().info(
            f"Step: dist={dist:.2f}, real_dist={real_dist:.2f}, "
            f"bearing={bearing:.2f}, visible={self._visible}, "
            f"v_cmd={v_cmd:.2f}, w_cmd={w_cmd:.2f}, "
            f"min_obs_depth={min_obs_depth:.2f}, "
            f"v_real={vx:.3f}, w_real={wz:.3f}, "
            f"reward={reward:.2f}, term={term}, trunc={trunc}, reason={reason}"
        )

        info = {
            "reason": reason,
            "robot_start": self._robot_start,
            "goal_start": self._goal_start,
            "robot_final": self._robot_last,
            "goal_final": self._goal_last,
            "min_dist": self._min_dist,
            "robot_traj": self._ep_robot_traj,
            "min_obstacle_depth": min_obs_depth,
        }
        return obs, float(reward), term, trunc, info

    def close(self):
        try:
            self.ros.send_cmd(0.0, 0.0)
        except Exception as e:
            print(f"Error sending stop cmd during close: {e}")

        try:
            if self.exec is not None:
                self.exec.shutdown()
        except Exception as e:
            print(f"Error shutting down executor: {e}")

        try:
            if self.ros is not None:
                self.ros.destroy_node()
        except Exception as e:
            print(f"Error destroying node: {e}")
