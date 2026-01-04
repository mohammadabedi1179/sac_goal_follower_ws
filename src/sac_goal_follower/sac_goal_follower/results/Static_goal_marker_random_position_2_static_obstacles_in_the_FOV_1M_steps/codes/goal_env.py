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
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import String, Float32

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
    - Subscribes to obstacle depth JSON (includes x_m, y_m in base_link frame).
    - Subscribes to ultrasonic distances (4 sensors, Float32 distance_m).
    - Publishes cmd_vel.
    """

    def __init__(
        self,
        cmd_topic: str,
        goal_state_topic: str,
        goal_odom_topic: str,
        robot_odom_topic: str,
        obstacle_topic: str = "/follower_robot/obstacles_depth",
        ultrasonic_topics: dict | None = None,
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

        # Ultrasonic buffers
        self.ultra = {
            "front_left": None,
            "front_right": None,
            "left_side": None,
            "right_side": None,
        }
        self.ultra_stamp = {k: 0.0 for k in self.ultra.keys()}

        # Subscriptions
        self.create_subscription(
            GoalMarkerState,
            goal_state_topic,
            self._goal_state_cb,
            10,
        )
        self.create_subscription(Odometry, goal_odom_topic, self._goal_odom_cb, 10)
        self.create_subscription(ModelStates, "/model_states", self._model_states_cb, 10)
        self.create_subscription(String, obstacle_topic, self._obstacle_cb, 10)

        if ultrasonic_topics is None:
            ultrasonic_topics = {
                "front_left":  "/follower_robot/ultrasonic_bridge/front_left/distance_m",
                "front_right": "/follower_robot/ultrasonic_bridge/front_right/distance_m",
                "left_side":   "/follower_robot/ultrasonic_bridge/left_side/distance_m",
                "right_side":  "/follower_robot/ultrasonic_bridge/right_side/distance_m",
            }

        for key, topic in ultrasonic_topics.items():
            self.create_subscription(
                Float32,
                topic,
                lambda msg, k=key: self._ultra_cb(k, msg),
                10,
            )

    # --- Callbacks ---

    def _goal_state_cb(self, msg: GoalMarkerState):
        self.goal_state = msg

    def _goal_odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self.goal_pose = (p.x, p.y)

    def _model_states_cb(self, msg: ModelStates):
        """
        Extract robot pose and velocity from Gazebo ground-truth.
        """
        try:
            i = msg.name.index("my_robot")   # MUST match Gazebo model name
        except ValueError:
            return  # model not yet in list

        # --- Position ---
        p = msg.pose[i].position

        # --- Orientation -> yaw ---
        o = msg.pose[i].orientation
        yaw = math.atan2(
            2.0 * (o.w * o.z + o.x * o.y),
            1.0 - 2.0 * (o.y * o.y + o.z * o.z),
        )

        self.robot_pose = (p.x, p.y, yaw)

        # --- Velocities (Gazebo truth) ---
        self._last_odom_vx = msg.twist[i].linear.x
        self._last_odom_wz = msg.twist[i].angular.z


    def _obstacle_cb(self, msg: String):
        # Raw JSON string from stereo_box_depth_from_disparity_IQR_EMA_synced (includes x_m, y_m)
        self.obstacles_json = msg.data

    # --- Helpers ---

    def send_cmd(self, v: float, w: float) -> None:
        tw = Twist()
        tw.linear.x = float(v)
        tw.angular.z = float(w)
        self.cmd_pub.publish(tw)

    def _ultra_cb(self, key: str, msg: Float32):
        v = float(msg.data)
        self.ultra[key] = v
        self.ultra_stamp[key] = time.time()


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
        c_time=0.1,
        c_dist=0.3,
        c_lost=0.1,
        R_goal=50.0,
    ):
        super().__init__()

        # --------- ACTION SPACE ----------
        self.v_max = 1.0     # [m/s]
        self.w_max = 1.0     # [rad/s]
        self.action_low = np.array([-1.0, -1.0], dtype=np.float32)
        self.action_high = np.array([1.0, 1.0], dtype=np.float32)
        self.action_space = spaces.Box(
            low=self.action_low,
            high=self.action_high,
            dtype=np.float32,
        )

        # Obstacle handling / defaults
        self.max_obstacle_depth = 10.0        # default "no obstacle" depth

        # Ultrasonic handling
        self.ultra_max_range = 1.0          # your sensor max_range looks like 1.0
        self.ultra_stale_sec = 0.5          # ignore too-old readings

        # --------- OBSERVATION SPACE ----------
        # 12D observation (yours) + 2D potential field = 12? Actually here it's 10D + pf2 = 12D.
        #
        # Final observation:
        # [goal_dist, goal_bearing,
        #  obs_x_m, obs_y_m, obs_depth_m,
        #  ultra_front_left, ultra_front_right, ultra_left_side, ultra_right_side,
        #  is_visible,
        #  pf_x, pf_y]
        #
        # pf_x, pf_y are unit vector components in base_link frame suggesting direction:
        # attraction to goal + repulsion from obstacle + pseudo repulsion from ultrasonic.
        self.observation_space = spaces.Box(
            low=np.array(
                [0.0, -math.pi,   -np.inf, -np.inf, 0.0,    0.0, 0.0, 0.0, 0.0, 0.0,   -1.0, -1.0],
                dtype=np.float32
            ),
            high=np.array(
                [np.inf, math.pi, np.inf,  np.inf,  self.max_obstacle_depth,
                 self.ultra_max_range, self.ultra_max_range, self.ultra_max_range, self.ultra_max_range,
                 1.0,  1.0, 1.0],
                dtype=np.float32
            ),
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
        self.c_angle = 0.5     # penalty on |bearing|
        self.c_progress = 3.0  # reward for reducing distance
        self.c_ctrl = 0.01     # control effort penalty

        # Action smoothing
        self.smooth_alpha = 1.0
        self._prev_v_cmd = 0.0
        self._prev_w_cmd = 0.0

        # Search-on-reset
        self.enable_search_on_reset = True
        self.search_angular_speed = 1.0  # [rad/s]

        # ROS wrapper
        self.ros = _ROS(
            cmd_topic=cmd_topic,
            goal_state_topic=goal_state_topic,
            goal_odom_topic=goal_odom_topic,
            robot_odom_topic=robot_odom_topic,
            obstacle_topic="/follower_robot/obstacles_depth",
            ultrasonic_topics={
                "front_left":  "/follower_robot/ultrasonic_bridge/front_left/distance_m",
                "front_right": "/follower_robot/ultrasonic_bridge/front_right/distance_m",
                "left_side":   "/follower_robot/ultrasonic_bridge/left_side/distance_m",
                "right_side":  "/follower_robot/ultrasonic_bridge/right_side/distance_m",
            },
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

        # Collision / penalty parameters
        self.obstacle_safe_radius = 1.5       # start penalizing here [m]
        self.obstacle_collision_radius = 0.7  # collision radius [m]
        self.c_obstacle = 0.5                 # penalty factor

        self.ultra_safe_radius = 0.5        # start penalizing when closer than this
        self.ultra_collision_radius = 0.2   # terminate if closer than this
        self.c_ultra = 0.5                  # penalty strength

        # Potential field parameters (tunable)
        self.pf_k_att = 1.0
        self.pf_k_rep = 1.2
        self.pf_r0_cam = 2.0
        self.pf_k_u = 1.0
        self.pf_r0_u = 0.7

        # last valid obs:
        # [goal_dist, goal_bearing, obs_x, obs_y, obs_depth, fl, fr, ls, rs, visible, pf_x, pf_y]
        self._last_obs_valid = np.array(
            [7.08, 0.0,
             self.max_obstacle_depth, 0.0, self.max_obstacle_depth,
             self.ultra_max_range, self.ultra_max_range, self.ultra_max_range, self.ultra_max_range,
             0.0,
             1.0, 0.0],
            dtype=np.float32
        )

    # --- ROS spin helper ---

    def _spin(self, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            self.exec.spin_once(timeout_sec=0.001)

    # --- Parse nearest obstacle local XY from obstacles_json ---

    def _compute_nearest_obstacle_local(self) -> tuple[float, float, float]:
        """
        Returns (x_m, y_m, depth_m) for the nearest valid obstacle in base_link frame.
        """
        if getattr(self.ros, "obstacles_json", None) is None:
            return (self.max_obstacle_depth, 0.0, self.max_obstacle_depth)

        try:
            objs = json.loads(self.ros.obstacles_json)
        except Exception:
            return (self.max_obstacle_depth, 0.0, self.max_obstacle_depth)

        if not isinstance(objs, list) or len(objs) == 0:
            return (self.max_obstacle_depth, 0.0, self.max_obstacle_depth)

        best = None  # (depth, x, y)
        for o in objs:
            try:
                d = float(o.get("depth_m", float("inf")))
                if not (math.isfinite(d) and d > 0.0):
                    continue

                if "x_m" in o and "y_m" in o:
                    x = float(o.get("x_m"))
                    y = float(o.get("y_m"))
                    if not (math.isfinite(x) and math.isfinite(y)):
                        continue
                else:
                    continue

                if best is None or d < best[0]:
                    best = (d, x, y)
            except Exception:
                continue

        if best is None:
            return (self.max_obstacle_depth, 0.0, self.max_obstacle_depth)

        d, x, y = best
        d = max(0.0, min(d, self.max_obstacle_depth))
        return (float(x), float(y), float(d))

    # --- Read ultrasonic 4 distances (and also return min for reward) ---

    def _ultra_vector(self) -> tuple[np.ndarray, float]:
        """
        Returns:
          - ultra_vec: np.array([front_left, front_right, left_side, right_side], float32)
          - ultra_min: float (min of values)
        """
        now = time.time()
        keys = ["front_left", "front_right", "left_side", "right_side"]
        vals = []
        for k in keys:
            v = self.ros.ultra.get(k, None)
            if v is None:
                vals.append(self.ultra_max_range)
                continue
            if (now - self.ros.ultra_stamp.get(k, 0.0)) > self.ultra_stale_sec:
                vals.append(self.ultra_max_range)
                continue
            if not math.isfinite(v):
                vals.append(self.ultra_max_range)
                continue
            v = max(0.0, min(float(v), self.ultra_max_range))
            vals.append(v)

        ultra_vec = np.array(vals, dtype=np.float32)
        ultra_min = float(np.min(ultra_vec)) if ultra_vec.size > 0 else float(self.ultra_max_range)
        return ultra_vec, ultra_min

    # --- Potential Field computation ---

    def _compute_potential_field(
        self,
        goal_bearing: float,
        obs_x: float,
        obs_y: float,
        obs_depth: float,
        ultra_fl: float,
        ultra_fr: float,
        ultra_ls: float,
        ultra_rs: float,
    ) -> tuple[float, float]:
        """
        Computes a unit potential field vector (pf_x, pf_y) in base_link frame.
        - Attractive: toward goal bearing
        - Repulsive: from nearest stereo obstacle (if within pf_r0_cam)
        - Repulsive: from ultrasonic readings using pseudo directions
        """
        # attractive (unit)
        Fx = self.pf_k_att * math.cos(goal_bearing)
        Fy = self.pf_k_att * math.sin(goal_bearing)

        # repulsive from stereo obstacle (use local x,y)
        r = math.hypot(obs_x, obs_y)
        if math.isfinite(r) and r > 1e-3 and r < self.pf_r0_cam:
            mag = self.pf_k_rep * (1.0 / r - 1.0 / self.pf_r0_cam) / (r * r)
            Fx += mag * (-obs_x / r)
            Fy += mag * (-obs_y / r)

        # repulsive from ultrasonic with pseudo directions
        dirs = {
            "front_left":  (1.0, 0.35, ultra_fl),
            "front_right": (1.0, -0.35, ultra_fr),
            "left_side":   (0.0, 1.0, ultra_ls),
            "right_side":  (0.0, -1.0, ultra_rs),
        }

        for _, (dx, dy, du) in dirs.items():
            du = float(du)
            if not math.isfinite(du):
                continue
            du = max(0.05, min(du, self.ultra_max_range))
            if du < self.pf_r0_u:
                mag = self.pf_k_u * (1.0 / du - 1.0 / self.pf_r0_u) / (du * du)
                Fx += mag * (-dx)
                Fy += mag * (-dy)

        # normalize
        n = math.hypot(Fx, Fy)
        if n > 1e-6:
            Fx /= n
            Fy /= n
        else:
            Fx, Fy = 1.0, 0.0

        # keep finite
        if not (math.isfinite(Fx) and math.isfinite(Fy)):
            Fx, Fy = 1.0, 0.0

        return float(Fx), float(Fy)

    # --- Observation construction ---

    def _obs(self, default: bool = False):
        """
        Build observation:
        [goal_dist, goal_bearing, obs_x_m, obs_y_m, obs_depth_m,
         ultra_fl, ultra_fr, ultra_ls, ultra_rs, is_visible,
         pf_x, pf_y]
        """
        if default:
            return self._last_obs_valid.copy()

        st = self.ros.goal_state
        self._visible = False
        obs = None

        # obstacle local XY + depth (from stereo node)
        obs_x, obs_y, obs_depth = self._compute_nearest_obstacle_local()

        # ultrasound 4 distances
        ultra_vec, _ultra_min = self._ultra_vector()
        ultra_fl, ultra_fr, ultra_ls, ultra_rs = map(float, ultra_vec.tolist())

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

            pf_x, pf_y = self._compute_potential_field(
                goal_bearing=b,
                obs_x=obs_x,
                obs_y=obs_y,
                obs_depth=obs_depth,
                ultra_fl=ultra_fl,
                ultra_fr=ultra_fr,
                ultra_ls=ultra_ls,
                ultra_rs=ultra_rs,
            )

            obs = np.array(
                [d, b, obs_x, obs_y, obs_depth, ultra_fl, ultra_fr, ultra_ls, ultra_rs, 1.0, pf_x, pf_y],
                np.float32
            )

        # CASE 2: not visible, but we have last known goal pose in world frame
        elif self._last_goal_pose is not None and self.ros.robot_pose is not None:
            rx, ry, ryaw = self.ros.robot_pose
            dx = self._last_goal_pose[0] - rx
            dy = self._last_goal_pose[1] - ry
            dist = math.sqrt(dx * dx + dy * dy)
            bearing = _wrap(math.atan2(dy, dx) - ryaw)

            pf_x, pf_y = self._compute_potential_field(
                goal_bearing=bearing,
                obs_x=obs_x,
                obs_y=obs_y,
                obs_depth=obs_depth,
                ultra_fl=ultra_fl,
                ultra_fr=ultra_fr,
                ultra_ls=ultra_ls,
                ultra_rs=ultra_rs,
            )

            obs = np.array(
                [dist, bearing, obs_x, obs_y, obs_depth, ultra_fl, ultra_fr, ultra_ls, ultra_rs, 0.0, pf_x, pf_y],
                np.float32
            )

        # CASE 3: fallback
        if obs is None:
            return self._last_obs_valid.copy()

        # Finite check
        if not np.all(np.isfinite(obs)):
            self.ros.get_logger().error(
                f"[OBS] Non-finite observation detected: {obs}, replacing with last valid obs."
            )
            return self._last_obs_valid.copy()

        self._last_obs_valid = obs
        return obs

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
            self.ros.get_logger().info(f"[{name}] Reset attempt {attempt + 1}/{max_attempts}")
            qz, qw = math.sin(yaw / 2.0), math.cos(yaw / 2.0)

            # include twist zeros to kill leftover momentum
            req = (
                "{state: {"
                f"name: '{name}', "
                "pose: {position: {"
                f"x: {x}, y: {y}, z: {z}"
                "}, orientation: {"
                f"z: {qz}, w: {qw}"
                "}}, "
                "twist: {"
                "linear: {x: 0.0, y: 0.0, z: 0.0}, "
                "angular: {x: 0.0, y: 0.0, z: 0.0}"
                "}"
                "}}"
            )

            cmd = [
                "ros2",
                "service",
                "call",
                "/set_entity_state",
                "gazebo_msgs/srv/SetEntityState",
                req,
            ]

            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if "success: True" in out.stdout or "success=True" in out.stdout:
                    self.ros.get_logger().info(f"[{name}] Reset confirmed by CLI")
                    return True
                else:
                    self.ros.get_logger().warn(
                        f"[{name}] Reset not confirmed.\nstdout:\n{out.stdout}\nstderr:\n{out.stderr}"
                    )
            except subprocess.TimeoutExpired:
                self.ros.get_logger().warn(f"[{name}] CLI call timed out")

        self.ros.get_logger().error(f"[{name}] Failed to reset after {max_attempts}")
        return False

    def _search_full_rotation(self) -> bool:
        """
        Rotate robot 360° at fixed angular speed, stop if goal becomes visible.
        """
        if self.ros.robot_pose is None:
            self.ros.get_logger().warn("[SEARCH] No robot odom available, skipping search.")
            return False

        _, _, yaw0 = self.ros.robot_pose
        target_rotation = 4.0 * math.pi
        w = self.search_angular_speed

        self.ros.get_logger().info(f"[SEARCH] Starting 360-degree search at {w:.2f} rad/s")

        accumulated = 0.0
        prev_yaw = yaw0
        seen_once = False

        while accumulated < target_rotation:
            self.ros.send_cmd(0.0, w)
            self._spin(self.dt)

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

    # --- Extra helpers for hard reset ---

    def _call_service_cli(self, service, srv_type, args="{}"):
        try:
            cmd = ["ros2", "service", "call", service, srv_type, args]
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=5.0)
            if out.returncode != 0:
                self.ros.get_logger().warn(
                    f"Service call failed: {' '.join(cmd)}\nstdout:\n{out.stdout}\nstderr:\n{out.stderr}"
                )
            return out.returncode == 0
        except subprocess.TimeoutExpired:
            self.ros.get_logger().warn(f"Service call timed out: {' '.join(cmd)}")
            return False

    def _hard_stop_robot(self, duration=0.5):
        t_end = time.time() + duration
        while time.time() < t_end:
            self.ros.send_cmd(0.0, 0.0)
            self._spin(0.05)

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

        # 0) Pause physics and HARD stop robot
        self._call_service_cli("/pause_physics", "std_srvs/srv/Empty", "{}")
        self._hard_stop_robot(duration=0.5)

        # 1) Reset robot pose
        self._reset_entity_with_retry("my_robot", 0.0, 0.0, 0.3, 0.7854)

        # 2) Random goal on circle
        radius = 7.0
        angle = self.np_random.uniform(0.0, 2.0 * math.pi)
        gx = radius * math.cos(angle)
        gy = radius * math.sin(angle)
        self._reset_entity_with_retry("goal_marker", gx, gy, 0.75, 0.0)

        # Let odom update
        self._spin(0.3)

        # 3) Reset / teleport obstacles
        subprocess.run(
            ["ros2", "service", "call", "/reset_random_obstacles", "std_srvs/srv/Trigger", "{}"],
            capture_output=True,
            text=True,
        )
        self.ros.get_logger().info("[ENV] Random obstacles reset/teleported")

        self._spin(0.3)

        # 4) Unpause physics
        self._call_service_cli("/unpause_physics", "std_srvs/srv/Empty", "{}")

        # 5) One more hard stop
        self._hard_stop_robot(duration=0.3)

        # 6) Search
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
            f"Episode reset completed: initial obs={obs}, goal_spawn=({gx:.2f}, {gy:.2f}), seen_in_search={seen}"
        )
        return obs, {}

    # --- Gym API: step ---

    def step(self, action):
        a = np.clip(action, self.action_low, self.action_high).astype(np.float32)
        if not np.all(np.isfinite(a)):
            self.ros.get_logger().error(f"[STEP] Non-finite action from policy: {a}, zeroing.")
            a = np.array([0.0, 0.0], dtype=np.float32)

        v_des = float(a[0]) * self.v_max
        w_des = float(a[1]) * self.w_max

        v_cmd = (1.0 - self.smooth_alpha) * self._prev_v_cmd + self.smooth_alpha * v_des
        w_cmd = (1.0 - self.smooth_alpha) * self._prev_w_cmd + self.smooth_alpha * w_des
        self._prev_v_cmd = v_cmd
        self._prev_w_cmd = w_cmd

        self.ros.send_cmd(v_cmd, w_cmd)
        self._spin(self.dt)

        obs = self._obs()
        if not np.all(np.isfinite(obs)):
            self.ros.get_logger().error(f"[STEP] Non-finite obs from _obs(): {obs}, using last valid.")
            obs = self._last_obs_valid.copy()

        (dist, bearing,
         obs_x, obs_y, obs_depth,
         ultra_fl, ultra_fr, ultra_ls, ultra_rs,
         is_visible,
         pf_x, pf_y) = map(float, obs)

        # Derived mins for safety
        min_ultra = float(min(ultra_fl, ultra_fr, ultra_ls, ultra_rs))
        min_obs_depth = float(obs_depth)

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
        reward += self.c_progress * progress

        if self._visible:
            reward += 0.2
            if dist < 2.5:
                reward -= self.c_angle * abs(bearing)

        term = False
        trunc = False
        reason = ""

        # Ultrasonic collision / penalties (short-range safety)
        if min_ultra < self.ultra_collision_radius:
            reward -= 2.0 * self.R_goal
            term = True
            reason = "Ultrasonic collision"
        elif min_ultra < self.ultra_safe_radius:
            reward -= self.c_ultra * ((self.ultra_safe_radius - min_ultra) ** 2)

        # Obstacle (stereo) collision / penalties
        if min_obs_depth < self.obstacle_collision_radius:
            reward -= 2.0 * self.R_goal
            term = True
            reason = "Collision with obstacle"
        elif min_obs_depth < self.obstacle_safe_radius:
            reward -= self.c_obstacle * ((self.obstacle_safe_radius - min_obs_depth) ** 2)

        # Success
        if not term and self._visible and dist <= self.success_radius:
            reward += self.R_goal
            term = True
            reason = "Reached goal"

        # Lost marker too long
        if (time.time() - self._last_seen) >= self.lost_timeout:
            reward -= 0.4 * self.R_goal
            term = True
            reason = "Lost marker timeout"

        # Time limit
        if (time.time() - self._t0) >= self.time_limit:
            trunc = True
            reward -= self.R_goal
            reason = "Time limit reached"

        self.ros.get_logger().info(
            f"Step: dist={dist:.2f}, real_dist={real_dist:.2f}, "
            f"bearing={bearing:.2f}, visible={self._visible}, "
            f"v_cmd={v_cmd:.2f}, w_cmd={w_cmd:.2f}, "
            f"obs_local=(x={obs_x:.2f}, y={obs_y:.2f}), obs_depth={min_obs_depth:.2f}, "
            f"ultra=[fl={ultra_fl:.2f}, fr={ultra_fr:.2f}, ls={ultra_ls:.2f}, rs={ultra_rs:.2f}], "
            f"pf=(x={pf_x:.2f}, y={pf_y:.2f}), "
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

            "nearest_obstacle_local_xy": (obs_x, obs_y),
            "nearest_obstacle_depth": min_obs_depth,
            "ultra_distances": {
                "front_left": ultra_fl,
                "front_right": ultra_fr,
                "left_side": ultra_ls,
                "right_side": ultra_rs,
            },
            "min_ultrasonic_distance": min_ultra,
            "potential_field": (pf_x, pf_y),
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
