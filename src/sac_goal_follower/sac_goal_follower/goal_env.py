#!/usr/bin/env python3
import time
import math
import json
import subprocess
from typing import Optional, Tuple, List, Dict, Any

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


def _yaw_from_quat(x, y, z, w) -> float:
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _world_from_local(rx, ry, ryaw, lx, ly) -> Tuple[float, float]:
    # base_link local (x forward, y left) -> world
    wx = rx + lx * math.cos(ryaw) - ly * math.sin(ryaw)
    wy = ry + lx * math.sin(ryaw) + ly * math.cos(ryaw)
    return wx, wy


def _local_from_world(rx, ry, ryaw, wx, wy) -> Tuple[float, float]:
    # world -> base_link local
    dx = wx - rx
    dy = wy - ry
    lx =  math.cos(ryaw) * dx + math.sin(ryaw) * dy
    ly = -math.sin(ryaw) * dx + math.cos(ryaw) * dy
    return lx, ly


class _ROS(Node):
    """
    Thin ROS2 wrapper for the SAC environment.

    - Subscribes to GoalMarkerState (vision + depth results).
    - Subscribes to goal odometry.
    - Subscribes to /model_states for robot ground-truth pose and velocity.
    - Subscribes to obstacle depth JSON (includes x_m, y_m in base_link frame).
    - Subscribes to ultrasonic distances (4 sensors, Float32 distance_m).
    - Publishes cmd_vel.
    """

    def __init__(
        self,
        cmd_topic: str,
        goal_state_topic: str,
        goal_odom_topic: str,
        obstacle_topic: str = "/follower_robot/obstacles_depth",
        ultrasonic_topics: dict | None = None,
        robot_model_name: str = "my_robot",
    ):
        super().__init__("sac_goal_env_node")

        self.robot_model_name = str(robot_model_name)

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)

        # Buffers
        self.goal_state: GoalMarkerState | None = None
        self.goal_pose = None                 # (x, y) from /goal_marker/odom
        self.robot_pose = None                # (x, y, yaw) from /model_states (Gazebo truth)
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
        self.create_subscription(GoalMarkerState, goal_state_topic, self._goal_state_cb, 10)
        self.create_subscription(Odometry, goal_odom_topic, self._goal_odom_cb, 10)

        # Gazebo truth for robot
        self.create_subscription(ModelStates, "/model_states", self._model_states_cb, 10)

        # Obstacles JSON from stereo node
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
        # Ground-truth obstacle poses from /model_states (world frame)
        self.gt_obstacle_names = ["yolo_obstacle_0", "yolo_obstacle_1", "yolo_obstacle_2"]
        self.gt_obstacles_world = {n: None for n in self.gt_obstacle_names}  # name -> (x,y)

    # --- Callbacks ---

    def _goal_state_cb(self, msg: GoalMarkerState):
        self.goal_state = msg

    def _goal_odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self.goal_pose = (float(p.x), float(p.y))

    def _model_states_cb(self, msg: ModelStates):
        """
        Extract robot pose and velocity from Gazebo ground-truth.
        """
        try:
            i = msg.name.index(self.robot_model_name)   # MUST match Gazebo model name
        except ValueError:
            return

        p = msg.pose[i].position
        o = msg.pose[i].orientation
        yaw = _yaw_from_quat(o.x, o.y, o.z, o.w)

        self.robot_pose = (float(p.x), float(p.y), float(yaw))

        self._last_odom_vx = float(msg.twist[i].linear.x)
        self._last_odom_wz = float(msg.twist[i].angular.z)
        # --- Ground-truth obstacles (Gazebo truth) ---
        for name in self.gt_obstacle_names:
            try:
                j = msg.name.index(name)
                p2 = msg.pose[j].position
                self.gt_obstacles_world[name] = (float(p2.x), float(p2.y))
            except ValueError:
                # keep previous (or None)
                pass


    def _obstacle_cb(self, msg: String):
        self.obstacles_json = msg.data

    # --- Helpers ---

    def send_cmd(self, v: float, w: float) -> None:
        tw = Twist()
        tw.linear.x = float(v)
        tw.angular.z = float(w)
        self.cmd_pub.publish(tw)

    def _ultra_cb(self, key: str, msg: Float32):
        self.ultra[key] = float(msg.data)
        self.ultra_stamp[key] = time.time()


class GoalFollowerEnv(Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        cmd_topic="/follower_robot/cmd_vel",
        goal_state_topic="/follower_robot/depth_cam/goal_marker_state",
        goal_odom_topic="/goal_marker/odom",
        wheel_radius=0.0925,
        wheel_separation=0.66108,
        dt=0.1,
        lost_timeout=25.0,
        success_radius=1.5,
        time_limit=80.0,
        c_time=0.1,
        c_dist=0.3,
        c_lost=0.1,
        R_goal=50.0,
        cam_x=0.4,
        cam_y=0.0,
    ):
        super().__init__()

        # --------- ACTION SPACE ----------
        self.v_max = 1.0
        self.w_max = 1.0
        self.action_low = np.array([-1.0, -1.0], dtype=np.float32)
        self.action_high = np.array([1.0, 1.0], dtype=np.float32)
        self.action_space = spaces.Box(low=self.action_low, high=self.action_high, dtype=np.float32)

        # Defaults
        self.max_obstacle_depth = 10.0

        # Ultrasonic
        self.ultra_max_range = 1.0
        self.ultra_stale_sec = 0.5

        # --------- OBSTACLE MEMORY (Pattern A) ----------
        self.num_obs_mem = 3
        self.obs_match_dist = 1.2          # meters in WORLD to be considered same obstacle
        self.obs_expire_s = 30.0           # after this, slot becomes invalid
        self.obs_slots: List[Dict[str, Any]] = [
            {"valid": False, "wx": 0.0, "wy": 0.0, "last_seen": 0.0} for _ in range(self.num_obs_mem)
        ]

        # --------- OBSERVATION SPACE (18D) ----------
        # [goal_dist, goal_bearing,
        #  x1,y1,d1, x2,y2,d2, x3,y3,d3,
        #  ultra_fl, ultra_fr, ultra_ls, ultra_rs,
        #  is_visible,
        #  pf_x, pf_y]
        low = np.array(
            [
                0.0, -math.pi,
                -np.inf, -np.inf, 0.0,
                -np.inf, -np.inf, 0.0,
                -np.inf, -np.inf, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0,
                -1.0, -1.0,
            ],
            dtype=np.float32
        )
        high = np.array(
            [
                np.inf, math.pi,
                np.inf, np.inf, self.max_obstacle_depth,
                np.inf, np.inf, self.max_obstacle_depth,
                np.inf, np.inf, self.max_obstacle_depth,
                self.ultra_max_range, self.ultra_max_range, self.ultra_max_range, self.ultra_max_range,
                1.0,
                1.0, 1.0,
            ],
            dtype=np.float32
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Kinematics
        self.r = wheel_radius
        self.L = wheel_separation

        self.dt = dt
        self.lost_timeout = lost_timeout
        self.success_radius = success_radius
        self.time_limit = time_limit

        self.cam_x = float(cam_x)
        self.cam_y = float(cam_y)

        # Reward parameters
        self.c_time = c_time
        self.c_dist = c_dist
        self.c_lost = c_lost
        self.R_goal = R_goal
        self.c_angle = 0.5
        self.c_progress = 3.0
        self.c_ctrl = 0.01

        # Action smoothing
        self.smooth_alpha = 1.0
        self._prev_v_cmd = 0.0
        self._prev_w_cmd = 0.0

        # Search-on-reset
        self.enable_search_on_reset = True
        self.search_angular_speed = 1.0

        # ROS wrapper
        self.ros = _ROS(
            cmd_topic=cmd_topic,
            goal_state_topic=goal_state_topic,
            goal_odom_topic=goal_odom_topic,
            obstacle_topic="/follower_robot/obstacles_depth",
            ultrasonic_topics={
                "front_left":  "/follower_robot/ultrasonic_bridge/front_left/distance_m",
                "front_right": "/follower_robot/ultrasonic_bridge/front_right/distance_m",
                "left_side":   "/follower_robot/ultrasonic_bridge/left_side/distance_m",
                "right_side":  "/follower_robot/ultrasonic_bridge/right_side/distance_m",
            },
            robot_model_name="my_robot",
        )
        self.exec = SingleThreadedExecutor()
        self.exec.add_node(self.ros)

        self._t0 = None
        self._last_seen = None
        self._last_goal_pose = None
        self._visible = False

        self._prev_dist = None

        # Logging
        self._ep_robot_traj = []
        self._robot_start = None
        self._goal_start = None
        self._robot_last = None
        self._goal_last = None
        self._min_dist = None

        # Collision / penalty parameters
        self.obstacle_safe_radius = 1.5
        self.obstacle_collision_radius = 0.7
        self.c_obstacle = 0.5

        self.ultra_safe_radius = 0.5
        self.ultra_collision_radius = 0.2
        self.c_ultra = 0.5

        # Potential field parameters
        self.pf_k_att = 1.0
        self.pf_k_rep = 1.2
        self.pf_r0_cam = 2.5
        self.pf_k_u = 1.0
        self.pf_r0_u = 0.7

        # Shaping
        self.progress_deadzone = 0.01
        self.stuck_penalty = 0.05
        self.visible_bonus = 0.2

        self.pf_align_gain = 0.25

        # last valid obs (18D)
        self._last_obs_valid = np.array(
            [
                7.0, 0.0,
                0.0, 0.0, self.max_obstacle_depth,
                0.0, 0.0, self.max_obstacle_depth,
                0.0, 0.0, self.max_obstacle_depth,
                self.ultra_max_range, self.ultra_max_range, self.ultra_max_range, self.ultra_max_range,
                0.0,
                1.0, 0.0
            ],
            dtype=np.float32
        )

    # --- ROS spin helper ---

    def _spin(self, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            self.exec.spin_once(timeout_sec=0.001)

    # --- Ultrasonic vector ---

    def _ultra_vector(self) -> Tuple[np.ndarray, float]:
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

    # --- Pattern A: obstacle parsing + memory update (WORLD) ---

    def _parse_obstacles_local(self) -> List[Tuple[float, float, float]]:
        """
        Returns list of (x_m, y_m, depth_m) from obstacle JSON.

        IMPORTANT:
        - x_m,y_m are assumed ALREADY in base_link frame (+x forward, +y left).
        - Your confirmed behavior: right image side => bearing>0 => y_m < 0.
          That is consistent with base_link +y left. So NO sign flip is applied here.
        """
        if getattr(self.ros, "obstacles_json", None) is None:
            return []
        try:
            objs = json.loads(self.ros.obstacles_json)
        except Exception:
            return []
        if not isinstance(objs, list):
            return []

        out = []
        for o in objs:
            try:
                if "x_m" not in o or "y_m" not in o:
                    continue
                x = float(o.get("x_m"))
                y = float(o.get("y_m"))
                d = float(o.get("depth_m", math.hypot(x, y)))
                if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(d)):
                    continue
                if d <= 0.0:
                    continue
                d = max(0.0, min(d, self.max_obstacle_depth))
                out.append((x, y, d))
            except Exception:
                continue
        return out

    def _expire_obstacle_slots(self, now: float):
        for s in self.obs_slots:
            if s["valid"] and (now - float(s["last_seen"])) > self.obs_expire_s:
                s["valid"] = False

    def _update_obstacle_memory(self):
        """
        Pattern A:
        - Convert current detections (local) -> WORLD using robot_pose.
        - Match each detection to existing slots by nearest WORLD distance (< obs_match_dist).
        - Unmatched detections fill empty slots, else replace stalest slot.
        - Slots keep WORLD pose even when not visible.
        """
        now = time.time()
        self._expire_obstacle_slots(now)

        if self.ros.robot_pose is None:
            return

        rx, ry, ryaw = self.ros.robot_pose

        dets_local = self._parse_obstacles_local()
        if not dets_local:
            return

        dets_world = []
        for (lx, ly, d) in dets_local:
            wx, wy = _world_from_local(rx, ry, ryaw, lx, ly)
            dets_world.append({"wx": wx, "wy": wy, "depth": d})

        used_slots = set()

        # greedy match by nearest slot
        for det in dets_world:
            best_j = None
            best_dist = float("inf")
            for j, s in enumerate(self.obs_slots):
                if not s["valid"]:
                    continue
                if j in used_slots:
                    continue
                dx = det["wx"] - float(s["wx"])
                dy = det["wy"] - float(s["wy"])
                dist = math.hypot(dx, dy)
                if dist < best_dist:
                    best_dist = dist
                    best_j = j

            if best_j is not None and best_dist <= self.obs_match_dist:
                s = self.obs_slots[best_j]
                s["wx"] = float(det["wx"])
                s["wy"] = float(det["wy"])
                s["last_seen"] = now
                s["valid"] = True
                used_slots.add(best_j)
                continue

            # no match -> put into empty slot or replace stalest
            empty_j = None
            for j, s in enumerate(self.obs_slots):
                if not s["valid"]:
                    empty_j = j
                    break

            if empty_j is not None:
                s = self.obs_slots[empty_j]
                s["wx"] = float(det["wx"])
                s["wy"] = float(det["wy"])
                s["last_seen"] = now
                s["valid"] = True
                used_slots.add(empty_j)
            else:
                stalest_j = int(np.argmin([float(s["last_seen"]) for s in self.obs_slots]))
                s = self.obs_slots[stalest_j]
                s["wx"] = float(det["wx"])
                s["wy"] = float(det["wy"])
                s["last_seen"] = now
                s["valid"] = True
                used_slots.add(stalest_j)

    def _slots_local_triplets_sorted_by_threat(self) -> List[Tuple[float, float, float]]:
        """
        Return 3 tuples (lx, ly, depth) computed from WORLD slots -> LOCAL,
        sorted by "threat" (closer first).
        If slot invalid, returns (0,0,max_depth).
        """
        if self.ros.robot_pose is None:
            return [(0.0, 0.0, self.max_obstacle_depth) for _ in range(self.num_obs_mem)]

        rx, ry, ryaw = self.ros.robot_pose

        triplets = []
        for s in self.obs_slots:
            if not s["valid"]:
                triplets.append((0.0, 0.0, self.max_obstacle_depth))
                continue

            lx, ly = _local_from_world(rx, ry, ryaw, float(s["wx"]), float(s["wy"]))
            d = math.hypot(lx, ly)
            d = max(0.0, min(d, self.max_obstacle_depth))
            triplets.append((float(lx), float(ly), float(d)))

        triplets_sorted = sorted(triplets, key=lambda t: t[2])
        return triplets_sorted[:3]

    # --- Potential Field computation (uses all 3 obstacles) ---

    def _compute_potential_field(
        self,
        goal_bearing: float,
        obs_triplets: List[Tuple[float, float, float]],
        ultra_fl: float,
        ultra_fr: float,
        ultra_ls: float,
        ultra_rs: float,
    ) -> Tuple[float, float]:
        # attractive
        Fx = self.pf_k_att * math.cos(goal_bearing)
        Fy = self.pf_k_att * math.sin(goal_bearing)

        # repulsive from up to 3 camera obstacles
        for (ox, oy, _od) in obs_triplets:
            r = math.hypot(ox, oy)
            if math.isfinite(r) and r > 1e-3 and r < self.pf_r0_cam:
                mag = self.pf_k_rep * (1.0 / r - 1.0 / self.pf_r0_cam) / (r * r)
                Fx += mag * (-ox / r)
                Fy += mag * (-oy / r)

        # repulsive from ultrasonic (pseudo directions)
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

        if not (math.isfinite(Fx) and math.isfinite(Fy)):
            Fx, Fy = 1.0, 0.0
        return float(Fx), float(Fy)

    # --- Observation construction ---

    def _obs(self, default: bool = False):
        """
        18D observation:
        [goal_dist, goal_bearing,
         x1,y1,d1, x2,y2,d2, x3,y3,d3,
         ultra_fl, ultra_fr, ultra_ls, ultra_rs,
         is_visible,
         pf_x, pf_y]
        """
        if default:
            return self._last_obs_valid.copy()

        # Update obstacle memory from latest JSON
        self._update_obstacle_memory()

        st = self.ros.goal_state
        self._visible = False
        obs = None

        ultra_vec, _ = self._ultra_vector()
        ultra_fl, ultra_fr, ultra_ls, ultra_rs = map(float, ultra_vec.tolist())

        # Get 3 obstacles, sorted by threat (closest first)
        obs_triplets = self._slots_local_triplets_sorted_by_threat()
        (x1, y1, d1), (x2, y2, d2), (x3, y3, d3) = obs_triplets

        # CASE 1: goal visible
        if st is not None and st.visible:
            self._visible = True
            self._last_seen = time.time()

            depth_cam = float(st.depth_m)

            # Goal marker bearing: camera RIGHT positive -> convert to yaw LEFT positive
            b_cam = float(st.bearing_rad)
            b_yaw = -b_cam
            b = _wrap(b_yaw)

            goal_dist = depth_cam

            # Estimate goal world pose from camera origin + depth
            if self.ros.robot_pose is not None and math.isfinite(depth_cam) and depth_cam > 0.0:
                rx, ry, ryaw = self.ros.robot_pose

                rx_cam = rx + self.cam_x * math.cos(ryaw) - self.cam_y * math.sin(ryaw)
                ry_cam = ry + self.cam_x * math.sin(ryaw) + self.cam_y * math.cos(ryaw)

                theta = ryaw + b_yaw
                gx = rx_cam + depth_cam * math.cos(theta)
                gy = ry_cam + depth_cam * math.sin(theta)
                self._last_goal_pose = (gx, gy)

                goal_dist = float(math.hypot(gx - rx, gy - ry))

            pf_x, pf_y = self._compute_potential_field(
                goal_bearing=b,
                obs_triplets=obs_triplets,
                ultra_fl=ultra_fl,
                ultra_fr=ultra_fr,
                ultra_ls=ultra_ls,
                ultra_rs=ultra_rs,
            )

            obs = np.array(
                [
                    goal_dist, b,
                    x1, y1, d1,
                    x2, y2, d2,
                    x3, y3, d3,
                    ultra_fl, ultra_fr, ultra_ls, ultra_rs,
                    1.0,
                    pf_x, pf_y
                ],
                dtype=np.float32
            )

        # CASE 2: goal not visible but we have last pose
        elif self._last_goal_pose is not None and self.ros.robot_pose is not None:
            rx, ry, ryaw = self.ros.robot_pose
            dx = self._last_goal_pose[0] - rx
            dy = self._last_goal_pose[1] - ry
            dist = float(math.hypot(dx, dy))
            bearing = _wrap(math.atan2(dy, dx) - ryaw)

            pf_x, pf_y = self._compute_potential_field(
                goal_bearing=bearing,
                obs_triplets=obs_triplets,
                ultra_fl=ultra_fl,
                ultra_fr=ultra_fr,
                ultra_ls=ultra_ls,
                ultra_rs=ultra_rs,
            )

            obs = np.array(
                [
                    dist, bearing,
                    x1, y1, d1,
                    x2, y2, d2,
                    x3, y3, d3,
                    ultra_fl, ultra_fr, ultra_ls, ultra_rs,
                    0.0,
                    pf_x, pf_y
                ],
                dtype=np.float32
            )

        if obs is None:
            return self._last_obs_valid.copy()

        if not np.all(np.isfinite(obs)):
            self.ros.get_logger().error(f"[OBS] Non-finite observation: {obs} -> using last valid.")
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

            cmd = ["ros2", "service", "call", "/set_entity_state", "gazebo_msgs/srv/SetEntityState", req]

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
        if self.ros.robot_pose is None:
            self.ros.get_logger().warn("[SEARCH] No robot pose available, skipping search.")
            return False

        _, _, yaw0 = self.ros.robot_pose
        target_rotation = 4.0 * math.pi
        w = self.search_angular_speed

        self.ros.get_logger().info(f"[SEARCH] Starting search at {w:.2f} rad/s")

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

        self.ros.get_logger().info(f"[SEARCH] Finished search; seen={seen_once}, rot={accumulated:.2f} rad")
        return seen_once

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

        # reset obstacle memory
        for s in self.obs_slots:
            s["valid"] = False
            s["wx"] = 0.0
            s["wy"] = 0.0
            s["last_seen"] = 0.0

        # Pause physics and hard stop
        self._call_service_cli("/pause_physics", "std_srvs/srv/Empty", "{}")
        self._hard_stop_robot(duration=0.5)

        # Reset robot
        self._reset_entity_with_retry("my_robot", 0.0, 0.0, 0.3, 0.7854)

        # Random goal
        radius = 7.0
        angle = self.np_random.uniform(0.0, 2.0 * math.pi)
        gx = radius * math.cos(angle)
        gy = radius * math.sin(angle)
        self._reset_entity_with_retry("goal_marker", gx, gy, 0.75, 0.0)

        self._spin(0.3)

        # Reset obstacles (your spawner)
        subprocess.run(
            ["ros2", "service", "call", "/reset_random_obstacles", "std_srvs/srv/Trigger", "{}"],
            capture_output=True,
            text=True,
        )
        self.ros.get_logger().info("[ENV] Random obstacles reset/teleported")

        self._spin(0.3)

        self._call_service_cli("/unpause_physics", "std_srvs/srv/Empty", "{}")
        self._hard_stop_robot(duration=0.3)

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
            f"Episode reset: obs={obs}, goal=({gx:.2f},{gy:.2f}), seen={seen}, "
            f"obs_match_dist={self.obs_match_dist:.2f}m, obs_expire_s={self.obs_expire_s:.1f}s"
        )
        return obs, {}

    # --- Gym API: step ---

    def step(self, action):
        a = np.clip(action, self.action_low, self.action_high).astype(np.float32)
        if not np.all(np.isfinite(a)):
            self.ros.get_logger().error(f"[STEP] Non-finite action: {a}, zeroing.")
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
            self.ros.get_logger().error(f"[STEP] Non-finite obs: {obs}, using last valid.")
            obs = self._last_obs_valid.copy()

        # unpack 18D
        dist = float(obs[0])
        bearing = float(obs[1])

        # obstacles triplets (3)
        o1 = (float(obs[2]), float(obs[3]), float(obs[4]))
        o2 = (float(obs[5]), float(obs[6]), float(obs[7]))
        o3 = (float(obs[8]), float(obs[9]), float(obs[10]))

        ultra_fl = float(obs[11])
        ultra_fr = float(obs[12])
        ultra_ls = float(obs[13])
        ultra_rs = float(obs[14])

        pf_x = float(obs[16])
        pf_y = float(obs[17])

        min_ultra = float(min(ultra_fl, ultra_fr, ultra_ls, ultra_rs))
        min_obs_depth = float(min(o1[2], o2[2], o3[2]))

        # Ground-truth distance for logging
        real_dist = float("inf")
        if self.ros.robot_pose is not None and self.ros.goal_pose is not None:
            rx, ry, _ = self.ros.robot_pose
            gx, gy = self.ros.goal_pose
            real_dist = math.hypot(gx - rx, gy - ry)

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

        # anti-stuck
        if abs(progress) < self.progress_deadzone:
            reward -= self.stuck_penalty

        # discourage backwards when goal not reached
        if self._visible and v_cmd < -0.05 and dist > self.success_radius * 2.0:
            reward -= 0.05

        # visibility shaping + bearing penalty near goal
        if self._visible:
            reward += self.visible_bonus
            if dist < 2.5:
                reward -= self.c_angle * abs(bearing)
        else:
            reward -= 0.05

        # potential-field alignment reward (heuristic)
        pf_v = np.array([pf_x, pf_y], dtype=np.float32)
        act_v = np.array(
            [max(-1.0, min(1.0, v_cmd / self.v_max)),
             max(-1.0, min(1.0, w_cmd / self.w_max))],
            dtype=np.float32
        )
        pf_n = float(np.linalg.norm(pf_v))
        act_n = float(np.linalg.norm(act_v))
        if pf_n > 1e-6 and act_n > 1e-6:
            align = float(np.dot(pf_v / pf_n, act_v / act_n))
            reward += self.pf_align_gain * align

        term = False
        trunc = False
        reason = ""

        # Ultrasonic safety
        if min_ultra < self.ultra_collision_radius:
            reward -= 1.2 * self.R_goal
            term = True
            reason = "Ultrasonic collision"
        elif min_ultra < self.ultra_safe_radius:
            reward -= self.c_ultra * ((self.ultra_safe_radius - min_ultra) ** 2)

        # Camera obstacle safety (based on closest stored obstacle)
        if min_obs_depth < self.obstacle_collision_radius:
            reward -= 1.2 * self.R_goal
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
            reward -= self.R_goal
            term = True
            reason = "Lost marker timeout"

        # Time limit
        if (time.time() - self._t0) >= self.time_limit:
            trunc = True
            reward -= self.R_goal
            reason = "Time limit reached"

        self.ros.get_logger().info(
            f"Step: dist={dist:.2f}, real_dist={real_dist:.2f}, bearing={bearing:.2f}, visible={self._visible}, "
            f"v_cmd={v_cmd:.2f}, w_cmd={w_cmd:.2f}, "
            f"obs1=(x={o1[0]:.2f},y={o1[1]:.2f},d={o1[2]:.2f}) "
            f"obs2=(x={o2[0]:.2f},y={o2[1]:.2f},d={o2[2]:.2f}) "
            f"obs3=(x={o3[0]:.2f},y={o3[1]:.2f},d={o3[2]:.2f}) "
            f"ultra=[{ultra_fl:.2f},{ultra_fr:.2f},{ultra_ls:.2f},{ultra_rs:.2f}] "
            f"pf=({pf_x:.2f},{pf_y:.2f}) v_real={vx:.3f} w_real={wz:.3f} "
            f"reward={reward:.2f} term={term} trunc={trunc} reason={reason}"
        )

        info = {
            "reason": reason,
            "robot_start": self._robot_start,
            "goal_start": self._goal_start,
            "robot_final": self._robot_last,
            "goal_final": self._goal_last,
            "min_dist": self._min_dist,
            "robot_traj": self._ep_robot_traj,

            "obstacles_local_sorted": [o1, o2, o3],
            "obstacles_world_slots": [
                {"valid": s["valid"], "wx": s["wx"], "wy": s["wy"], "last_seen": s["last_seen"]}
                for s in self.obs_slots
            ],
            "min_obstacle_depth": min_obs_depth,
            "ultra_distances": {
                "front_left": ultra_fl,
                "front_right": ultra_fr,
                "left_side": ultra_ls,
                "right_side": ultra_rs,
            },
            "min_ultrasonic_distance": min_ultra,
            "potential_field": (pf_x, pf_y),
            "cam_offset_base": (self.cam_x, self.cam_y),
            "last_goal_pose_est_world": self._last_goal_pose,
        }
        # --- Per-step debug payload for trajectory logging ---
        # Robot + goal world pose each step (GT)
        step_robot_pose = self.ros.robot_pose  # (x,y,yaw) or None
        step_goal_pose = self.ros.goal_pose    # (x,y) or None

        # GT obstacles world (name -> (x,y) or None)
        step_gt_obstacles = getattr(self.ros, "gt_obstacles_world", None)

        # Obstacles seen by perception this step (local base_link list)
        # This is what your robot "sees" at this timestep.
        seen_local = self._parse_obstacles_local()  # [(x,y,depth), ...]

        # Also provide the seen obstacles in world frame (if robot pose available)
        seen_world = []
        if step_robot_pose is not None:
            rx, ry, ryaw = step_robot_pose
            for (lx, ly, d) in seen_local:
                wx, wy = _world_from_local(rx, ry, ryaw, lx, ly)
                seen_world.append((float(wx), float(wy), float(d)))

        info["step_robot_pose"] = step_robot_pose
        info["step_goal_pose"] = step_goal_pose
        info["step_gt_obstacles_world"] = step_gt_obstacles
        info["step_seen_obstacles_local"] = seen_local
        info["step_seen_obstacles_world"] = seen_world

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
