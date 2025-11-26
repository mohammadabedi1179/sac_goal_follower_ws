import time
import math
import numpy as np
import subprocess

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

from gymnasium import Env, spaces
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from detectors_msgs.msg import GoalMarkerState  # high-level marker state


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
    - Publishes cmd_vel.
    """

    def __init__(
        self,
        cmd_topic: str,
        goal_state_topic: str,
        goal_odom_topic: str,
        robot_odom_topic: str,
    ):
        super().__init__("sac_goal_env_node")

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)

        # Buffers
        self.goal_state: GoalMarkerState | None = None
        self.goal_pose = None  # (x, y)
        self.robot_pose = None  # (x, y, yaw)

        # Subscriptions
        self.create_subscription(
            GoalMarkerState,
            goal_state_topic,
            self._goal_state_cb,
            10,
        )
        self.create_subscription(Odometry, goal_odom_topic, self._goal_odom_cb, 10)
        self.create_subscription(
            Odometry, robot_odom_topic, self._robot_odom_cb, 10
        )

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

        # --------- ACTION SPACE (FIXED RANGE) ----------
        # SAC will output actions in [-1, 1]. We map them to
        # physical linear and angular velocities.
        self.v_max = 2.0        # [m/s] ~ 2.5 km/h (safe)
        self.w_max = 2.0        # [rad/s]
        self.action_low = np.array([-1.0, -1.0], dtype=np.float32)
        self.action_high = np.array([1.0, 1.0], dtype=np.float32)

        self.action_space = spaces.Box(
            low=self.action_low,
            high=self.action_high,
            dtype=np.float32,
        )

        # Observation: [distance_to_goal, bearing]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -math.pi], np.float32),
            high=np.array([np.inf, math.pi], np.float32),
        )

        # Kinematic params kept if needed later
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

        # Extra reward shaping
        self.c_angle = 0.05     # penalty on |bearing|
        self.c_progress = 0.5   # reward for reducing distance
        self.c_ctrl = 0.01      # control effort penalty

        # Action smoothing (low-pass filter on v, w)
        self.smooth_alpha = 0.6   # 0 -> no movement, 1 -> no smoothing
        self._prev_v_cmd = 0.0
        self._prev_w_cmd = 0.0

        # --- Search parameters (for 360° search on reset) ---
        self.enable_search_on_reset = True
        self.search_angular_speed = 4.0  # [rad/s], can tune later

        self.ros = _ROS(
            cmd_topic=cmd_topic,
            goal_state_topic=goal_state_topic,
            goal_odom_topic=goal_odom_topic,
            robot_odom_topic=robot_odom_topic,
        )
        self.exec = SingleThreadedExecutor()
        self.exec.add_node(self.ros)

        self._t0 = None
        self._last_seen = None
        self._last_obs_valid = np.array([7.08, 0.0], np.float32)
        self._last_goal_pose = None
        self._visible = False

        # For progress-based reward
        self._prev_dist = None

        # --- NEW: per-episode logging buffers ---
        self._ep_robot_traj = []
        self._robot_start = None
        self._goal_start = None
        self._robot_last = None
        self._goal_last = None
        self._min_dist = None


    # --- ROS spin helper ---

    def _spin(self, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            self.exec.spin_once(timeout_sec=0.001)

    # --- Observation construction ---

    def _obs(self, default: bool = False):
        """
        Build observation [distance, bearing].

        If default=True (used at reset), just return last valid obs (no new state).
        """
        if default:
            return self._last_obs_valid.copy()

        st = self.ros.goal_state

        # Assume not visible by default; we will set True below if needed
        self._visible = False

        # ---- CASE 1: directly visible from perception node ----
        if st is not None and st.visible:
            self._visible = True
            self._last_seen = time.time()

            d = float(st.depth_m)
            b = _wrap(float(st.bearing_rad))

            # Estimate world-frame goal pose from robot pose
            if self.ros.robot_pose is not None:
                rx, ry, ryaw = self.ros.robot_pose
                gx = rx + d * math.cos(ryaw + b)
                gy = ry + d * math.sin(ryaw + b)
                self._last_goal_pose = (gx, gy)

            obs = np.array([d, b], np.float32)
            self._last_obs_valid = obs
            return obs

        # ---- CASE 2: not visible; propagate last known world pose using odom ----
        if self._last_goal_pose is not None and self.ros.robot_pose is not None:
            rx, ry, ryaw = self.ros.robot_pose
            dx = self._last_goal_pose[0] - rx
            dy = self._last_goal_pose[1] - ry
            dist = math.sqrt(dx * dx + dy * dy)
            bearing = _wrap(math.atan2(dy, dx) - ryaw)
            obs = np.array([dist, bearing], np.float32)
            self._last_obs_valid = obs
            return obs

        # ---- CASE 3: nothing better, return last valid obs ----
        return self._last_obs_valid.copy()

    # --- Gazebo reset helper (CLI, robust) ---

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
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=5,
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
        Rotate the robot exactly 360 degrees at a fixed angular speed.
        Stop immediately if the marker becomes visible.

        Returns:
            True  -> marker seen at least once
            False -> completed full rotation without seeing the marker
        """
        if self.ros.robot_pose is None:
            self.ros.get_logger().warn("[SEARCH] No robot odom available, skipping search.")
            return False

        # Store initial yaw
        _, _, yaw0 = self.ros.robot_pose

        target_rotation = 2.0 * math.pi     # 360 degrees
        w = self.search_angular_speed       # constant rotation speed

        self.ros.get_logger().info(
            f"[SEARCH] Starting 360-degree search at {w:.2f} rad/s"
        )

        accumulated = 0.0
        prev_yaw = yaw0
        seen_once = False

        while accumulated < target_rotation:
            # Command rotation
            self.ros.send_cmd(0.0, w)
            self._spin(self.dt)

            # Check visibility
            obs = self._obs(default=False)
            if self._visible:
                self.ros.get_logger().info(
                    f"[SEARCH] Goal seen during search (dist={obs[0]:.2f}, bearing={obs[1]:.2f})"
                )
                seen_once = True
                break

            # Update accumulated rotation based on yaw change
            if self.ros.robot_pose is not None:
                _, _, yaw = self.ros.robot_pose
                dyaw = _wrap(yaw - prev_yaw)
                accumulated += abs(dyaw)
                prev_yaw = yaw

        # Stop rotation
        self.ros.send_cmd(0.0, 0.0)
        self._spin(0.1)

        self.ros.get_logger().info(
            f"[SEARCH] Finished search; seen={seen_once}, total_rot={accumulated:.2f} rad"
        )

        return seen_once

    # --- Gym API: reset ---

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Do NOT start episode timer yet; we will start it after search.
        self._t0 = None
        self._last_seen = None
        self._visible = False
        self._last_goal_pose = None

        # Per-episode logging / trajectory buffers
        self._ep_robot_traj = []
        self._robot_start = None
        self._goal_start = None
        self._robot_last = None
        self._goal_last = None
        self._min_dist = None

        # Stop robot & reset smoothing state
        self._prev_v_cmd = 0.0
        self._prev_w_cmd = 0.0

        # Reset previous distance for progress-based reward
        self._prev_dist = None

        self.ros.send_cmd(0.0, 0.0)

        # Reset robot and goal_marker in Gazebo
        # Robot stays at origin
        self._reset_entity_with_retry("my_robot", 0.0, 0.0, 0.3, 0.7854)

        # --- Random goal on a circle of radius 7 m around origin ---
        radius = 7.0  # meters
        angle = self.np_random.uniform(0.0, 2.0 * math.pi)  # Gym's RNG
        gx = radius * math.cos(angle)
        gy = radius * math.sin(angle)
        self._reset_entity_with_retry("goal_marker", gx, gy, 0.75, 0.0)

        # Let things settle
        self._spin(0.5)

        # --- NEW: perform a 360-degree search until marker is visible ---
        seen = False
        if self.enable_search_on_reset:
            seen = self._search_full_rotation()

        # Now start episode time AFTER search
        now = time.time()
        self._t0 = now
        self._last_seen = now  # baseline; will be updated when visible

        # Initialize start poses if odom is available (after spawn + search)
        if self.ros.robot_pose is not None:
            rx, ry, _ = self.ros.robot_pose
            self._robot_start = (rx, ry)
            self._robot_last = (rx, ry)
            self._ep_robot_traj.append((rx, ry))

        if self.ros.goal_pose is not None:
            gx0, gy0 = self.ros.goal_pose
            self._goal_start = (gx0, gy0)
            self._goal_last = (gx0, gy0)

        # Build initial observation:
        #  - if search saw the goal, use fresh obs from sensors
        #  - if not, fallback to last_valid (default=True)
        obs = self._obs(default=not seen)

        # Initialize _prev_dist from initial observation
        self._prev_dist = float(obs[0])

        self.ros.get_logger().info(
            f"Episode reset completed: initial obs={obs}, "
            f"goal_spawn=({gx:.2f}, {gy:.2f}), seen_in_search={seen}"
        )
        return obs, {}


    # --- Gym API: step ---

    def step(self, action):
        # ---- ACTION RANGE & SMOOTHING ----
        # 1) Clip SAC outputs to [-1, 1]
        a = np.clip(action, self.action_low, self.action_high).astype(np.float32)
        a_v, a_w = float(a[0]), float(a[1])

        # 2) Map to physical velocities
        v_des = a_v * self.v_max
        w_des = a_w * self.w_max

        # 3) Smooth commands
        v_cmd = (1.0 - self.smooth_alpha) * self._prev_v_cmd + self.smooth_alpha * v_des
        w_cmd = (1.0 - self.smooth_alpha) * self._prev_w_cmd + self.smooth_alpha * w_des

        self._prev_v_cmd = v_cmd
        self._prev_w_cmd = w_cmd

        # 4) Send to robot
        self.ros.send_cmd(v_cmd, w_cmd)

        # Step ROS
        self._spin(self.dt)

        # Build observation
        obs = self._obs()
        dist, bearing = float(obs[0]), float(obs[1])

        # Ground-truth distance from odom (for logging)
        real_dist = float("inf")
        if self.ros.robot_pose is not None and self.ros.goal_pose is not None:
            rx, ry, _ = self.ros.robot_pose
            gx, gy = self.ros.goal_pose
            real_dist = math.sqrt((gx - rx) ** 2 + (gy - ry) ** 2)
        # Get true velocity
        vx = (
            self.ros._last_odom_vx
            if hasattr(self.ros, '_last_odom_vx')
            else None
        )
        wz = (
            self.ros._last_odom_wz
            if hasattr(self.ros, '_last_odom_wz')
            else None
        )
        # --- NEW: record robot pose & goal pose for trajectory logging ---
        if self.ros.robot_pose is not None:
            rx, ry, _ = self.ros.robot_pose
            self._robot_last = (rx, ry)
            self._ep_robot_traj.append((rx, ry))
        if self.ros.goal_pose is not None:
            gx, gy = self.ros.goal_pose
            self._goal_last = (gx, gy)

        # Update min distance seen so far (based on observation distance)
        if self._min_dist is None or dist < self._min_dist:
            self._min_dist = dist
        
        self.ros.get_logger().info(
            f"[STEP ACTUAL] v_real={vx:.3f}, w_real={wz:.3f}"
        )


        # --------- REWARD SHAPING (FIXED) ----------
        # Progress term: positive if distance is reduced
        if self._prev_dist is None:
            prev_dist = dist
        else:
            prev_dist = self._prev_dist

        progress = prev_dist - dist  # >0 if moving closer
        self._prev_dist = dist

        reward = 0.0

        # Small time penalty to encourage faster completion
        reward -= self.c_time * self.dt

        if self._visible:
            # Reward for getting closer
            reward += self.c_progress * progress

            # Penalize being far
            reward -= self.c_dist * dist

            # Penalize large bearing
            reward -= self.c_angle * abs(bearing)
        else:
            # If marker is not visible, penalize per unit time
            reward -= self.c_lost * self.dt

        # Penalize large control effort (encourage smooth / small motions)
        reward -= self.c_ctrl * (abs(v_cmd) + abs(w_cmd))

        term = False
        trunc = False
        reason = ""

        # Success: close enough to the goal while visible
        if self._visible and dist <= self.success_radius:
            reward += self.R_goal
            term = True
            reason = "Reached goal"

        # Failure: lost marker for too long
        if (time.time() - self._last_seen) >= self.lost_timeout:
            reward -= self.R_goal  # strong negative for losing the goal
            term = True
            reason = "Lost marker timeout"

        # Truncation: episode time limit
        if (time.time() - self._t0) >= self.time_limit:
            trunc = True
            reason = "Time limit reached"

        self.ros.get_logger().info(
            f"Step: dist={dist:.2f}, real_dist={real_dist:.2f}, "
            f"bearing={bearing:.2f}, visible={self._visible}, "
            f"v_cmd={v_cmd:.2f}, w_cmd={w_cmd:.2f}, "
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
        }
        return obs, float(reward), term, trunc, info

    def close(self):
        try:
            # Try to stop the robot nicely
            self.ros.send_cmd(0.0, 0.0)
        except Exception as e:
            print(f"Error sending stop cmd during close: {e}")

        try:
            # Shutdown executor first, then destroy node
            if self.exec is not None:
                self.exec.shutdown()
        except Exception as e:
            print(f"Error shutting down executor: {e}")

        try:
            if self.ros is not None:
                self.ros.destroy_node()
        except Exception as e:
            print(f"Error destroying node: {e}")


