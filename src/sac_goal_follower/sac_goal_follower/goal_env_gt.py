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
    wx = rx + lx * math.cos(ryaw) - ly * math.sin(ryaw)
    wy = ry + lx * math.sin(ryaw) + ly * math.cos(ryaw)
    return wx, wy


def _local_from_world(rx, ry, ryaw, wx, wy) -> Tuple[float, float]:
    dx = wx - rx
    dy = wy - ry
    lx =  math.cos(ryaw) * dx + math.sin(ryaw) * dy
    ly = -math.sin(ryaw) * dx + math.cos(ryaw) * dy
    return lx, ly


class _ROS(Node):
    """
    Thin ROS2 wrapper for the SAC environment.

    - Subscribes to GoalMarkerState (kept, but reward no longer depends on visibility).
    - Subscribes to goal odometry (kept).
    - Subscribes to /model_states for robot GT pose+velocity, goal pose+velocity, obstacle poses+velocities.
    - Subscribes to obstacle JSON (kept, debug).
    - Subscribes to ultrasonic distances (4 sensors).
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
        goal_model_name: str = "goal_marker",
        gt_obstacle_names: Optional[List[str]] = None,
    ):
        super().__init__("sac_goal_env_node")

        self.robot_model_name = str(robot_model_name)
        self.goal_model_name = str(goal_model_name)

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)

        # Buffers
        self.goal_state: GoalMarkerState | None = None
        self.goal_pose = None                 # (x, y) from /goal_marker/odom (kept)
        self.robot_pose = None                # (x, y, yaw) from /model_states (Gazebo truth)
        self._last_odom_vx = 0.0
        self._last_odom_wz = 0.0
        self.obstacles_json: str | None = None

        # GT goal pose and velocity from /model_states
        self.goal_pose_gt = None              # (x, y) from /model_states
        self.goal_vel_gt = None               # (vx, vy) from /model_states

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

        # GT obstacle poses and velocities from /model_states (world frame)
        if gt_obstacle_names is None:
            gt_obstacle_names = ["yolo_obstacle_0", "yolo_obstacle_1", "yolo_obstacle_2"]
        self.gt_obstacle_names = [str(x) for x in gt_obstacle_names]
        self.gt_obstacles_world = {n: None for n in self.gt_obstacle_names}  # name -> (x,y)
        self.gt_obstacles_vel = {n: None for n in self.gt_obstacle_names}    # name -> (vx,vy)

    # --- Callbacks ---
    def _goal_state_cb(self, msg: GoalMarkerState):
        self.goal_state = msg

    def _goal_odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self.goal_pose = (float(p.x), float(p.y))

    def _model_states_cb(self, msg: ModelStates):
        try:
            i = msg.name.index(self.robot_model_name)
        except ValueError:
            return

        p = msg.pose[i].position
        o = msg.pose[i].orientation
        yaw = _yaw_from_quat(o.x, o.y, o.z, o.w)

        self.robot_pose = (float(p.x), float(p.y), float(yaw))
        self._last_odom_vx = float(msg.twist[i].linear.x)
        self._last_odom_wz = float(msg.twist[i].angular.z)

        # Goal pose and velocity
        try:
            gi = msg.name.index(self.goal_model_name)
            pg = msg.pose[gi].position
            vg = msg.twist[gi].linear
            self.goal_pose_gt = (float(pg.x), float(pg.y))
            self.goal_vel_gt = (float(vg.x), float(vg.y))
        except ValueError:
            pass

        # Obstacles pose and velocity
        for name in self.gt_obstacle_names:
            try:
                j = msg.name.index(name)
                p2 = msg.pose[j].position
                v2 = msg.twist[j].linear
                self.gt_obstacles_world[name] = (float(p2.x), float(p2.y))
                self.gt_obstacles_vel[name] = (float(v2.x), float(v2.y))
            except ValueError:
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

    # ---------- normalization constants ----------
    GOAL_DIST_MAX = 15.0           # max expected goal distance
    OBS_COORD_MAX = 10.0           # max expected obstacle local coordinate
    VEL_MAX = 2.0                  # max expected velocity (for goal and obstacles)

    def __init__(
        self,
        cmd_topic="/follower_robot/cmd_vel",
        goal_state_topic="/follower_robot/depth_cam/goal_marker_state",
        goal_odom_topic="/goal_marker/odom",
        wheel_radius=0.0925,
        wheel_separation=0.66108,
        dt=0.1,
        success_radius=1.5,
        time_limit=80.0,
        # ---- FIX #1: Reward scale ----
        R_goal=50.0,                      # was 1.0
        R_collision=-50.0,                # was -1.0
        gamma_shaping=0.97,
        step_penalty=0.005,               # was 0.01
        ttc_threshold=2.0,
        cam_x=0.4,
        cam_y=0.0,
        use_ground_truth_geometry: bool = True,
        goal_model_name: str = "goal_marker",
        gt_obstacle_names: Optional[List[str]] = None,
        # ---- FIX #2: Action smoothing ----
        smooth_alpha: float = 0.70,       # was 0.20
        # ---- FIX #4: TTC lateral filter width ----
        ttc_lateral_margin: float = 0.8,
    ):
        super().__init__()

        self.use_ground_truth_geometry = bool(use_ground_truth_geometry)

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

        # --------- OBSTACLE MEMORY ----------
        self.num_obs_mem = 3
        self.obs_match_dist = 1.2
        self.obs_expire_s = 30.0
        self.obs_slots: List[Dict[str, Any]] = [
            {"valid": False, "wx": 0.0, "wy": 0.0, "vx": 0.0, "vy": 0.0, "last_seen": 0.0} 
            for _ in range(self.num_obs_mem)
        ]

        # --------- OBSERVATION SPACE (27D) ----------
        # [goal_dist, goal_bearing, goal_rel_vx, goal_rel_vy, v, w,
        #  obs1_x, obs1_y, obs1_d, obs1_rel_vx, obs1_rel_vy,
        #  obs2_x, obs2_y, obs2_d, obs2_rel_vx, obs2_rel_vy,
        #  obs3_x, obs3_y, obs3_d, obs3_rel_vx, obs3_rel_vy,
        #  ultra_fl, ultra_fr, ultra_ls, ultra_rs,
        #  prev_v_cmd, prev_w_cmd]
        #
        # ALL features normalized to roughly [-1, 1]
        low = np.array(
            [
                0.0, -1.0,                        # goal_dist_norm, bearing_norm
                -1.0, -1.0,                       # goal_rel_vx_norm, goal_rel_vy_norm
                -1.0, -1.0,                       # v_norm, w_norm
                -1.0, -1.0, 0.0, -1.0, -1.0,      # obs1 x,y,d,rel_vx,rel_vy normalized
                -1.0, -1.0, 0.0, -1.0, -1.0,      # obs2
                -1.0, -1.0, 0.0, -1.0, -1.0,      # obs3
                0.0, 0.0, 0.0, 0.0,               # ultrasonics (already 0-1)
                -1.0, -1.0,                       # prev_v_cmd, prev_w_cmd
            ],
            dtype=np.float32
        )
        high = np.array(
            [
                1.0, 1.0,
                1.0, 1.0,
                1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0,
                1.0, 1.0,
            ],
            dtype=np.float32
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Kinematics
        self.r = wheel_radius
        self.L = wheel_separation
        self.dt = dt

        self.success_radius = success_radius
        self.time_limit = time_limit

        self.cam_x = float(cam_x)
        self.cam_y = float(cam_y)

        # Reward parameters
        self.R_goal = float(R_goal)
        self.R_collision = float(R_collision)
        self.gamma = float(gamma_shaping)
        self.step_penalty = float(step_penalty)
        self.ttc_threshold = float(ttc_threshold)

        # Action smoothing (FIX #2)
        self.smooth_alpha = float(np.clip(smooth_alpha, 0.0, 1.0))
        self._prev_v_cmd = 0.0
        self._prev_w_cmd = 0.0
        self.c_progress = 2.0              # was 3.0 (FIX #1: reduce to not overshadow terminal)

        # FIX #2: action-change penalty coefficient (smoothness via reward, not filter)
        self.c_action_change = 0.3

        # FIX #4: TTC lateral margin
        self.ttc_lateral_margin = float(ttc_lateral_margin)

        # FIX #9: ultrasonic proximity penalty zone
        self.ultra_proximity_zone = 0.5    # start penalizing below this distance
        self.c_ultra_proximity = 2.0       # penalty coefficient

        # FIX #10-green: bearing alignment reward
        self.c_bearing = 0.3

        # Search-on-reset (kept)
        self.enable_search_on_reset = True
        self.search_angular_speed = 1.0

        # Collision thresholds (kept)
        self.obstacle_collision_radius = 0.7
        self.ultra_collision_radius = 0.2

        # --------- SAFETY SHIELD ----------
        self.shield_enabled = True
        # Zone thresholds (ultrasonic-based, in meters)
        self.shield_critical_dist = 0.25     # CRITICAL: emergency full stop
        self.shield_danger_dist = 0.50       # DANGER: strong deceleration + steer away
        self.shield_caution_dist = 0.80      # CAUTION: cap forward speed
        # Zone thresholds (camera obstacle depth, in meters)
        self.shield_obs_critical = 0.9       # ~obstacle_collision_radius + margin
        self.shield_obs_danger = 1.5
        self.shield_obs_caution = 2.5
        # Speed limits per zone
        self.shield_caution_max_v = 0.4      # max forward speed in caution zone
        self.shield_danger_max_v = 0.1       # max forward speed in danger zone
        self.shield_danger_steer_bias = 0.5  # angular bias away from closest obstacle
        # Tracking
        self._shield_interventions = 0       # count per episode
        self._shield_active = False          # was shield active this step?

        # ROS wrapper
        if gt_obstacle_names is None:
            gt_obstacle_names = ["yolo_obstacle_0", "yolo_obstacle_1", "yolo_obstacle_2"]

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
            goal_model_name=goal_model_name,
            gt_obstacle_names=gt_obstacle_names,
        )
        self.exec = SingleThreadedExecutor()
        self.exec.add_node(self.ros)

        # Episode state
        self._t0 = None
        self._prev_phi = 0.0

        # Logging
        self._ep_robot_traj = []
        self._robot_start = None
        self._goal_start = None
        self._robot_last = None
        self._goal_last = None
        self._min_dist = None

        # last obs fallback (normalized, 27D)
        self._last_obs_valid = np.array(
            [
                7.0 / self.GOAL_DIST_MAX, 0.0,
                0.0, 0.0,
                0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0,
                1.0, 1.0, 1.0, 1.0,
                0.0, 0.0,
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

    # --- Perception obstacle parsing (kept for debug) ---
    def _parse_obstacles_local(self) -> List[Tuple[float, float, float]]:
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

    # GT obstacle memory update with velocities
    def _update_obstacle_memory_gt(self):
        now = time.time()
        self._expire_obstacle_slots(now)

        if self.ros.robot_pose is None:
            return

        rx, ry, _ = self.ros.robot_pose

        pts = []
        for name in self.ros.gt_obstacle_names:
            p = self.ros.gt_obstacles_world.get(name, None)
            v = self.ros.gt_obstacles_vel.get(name, None)
            if p is None:
                continue
            # Store position and velocity
            vx, vy = (float(v[0]), float(v[1])) if v is not None else (0.0, 0.0)
            pts.append((float(p[0]), float(p[1]), vx, vy))

        pts.sort(key=lambda item: math.hypot(item[0] - rx, item[1] - ry))

        for j in range(self.num_obs_mem):
            if j < len(pts):
                wx, wy, vx, vy = pts[j]
                self.obs_slots[j]["valid"] = True
                self.obs_slots[j]["wx"] = wx
                self.obs_slots[j]["wy"] = wy
                self.obs_slots[j]["vx"] = vx
                self.obs_slots[j]["vy"] = vy
                self.obs_slots[j]["last_seen"] = now
            else:
                self.obs_slots[j]["valid"] = False

    def _slots_local_quintuplets_sorted_by_threat(self) -> List[Tuple[float, float, float, float, float]]:
        """Returns (lx, ly, d, rel_vx, rel_vy) for each obstacle slot"""
        if self.ros.robot_pose is None:
            return [(0.0, 0.0, self.max_obstacle_depth, 0.0, 0.0) for _ in range(self.num_obs_mem)]

        rx, ry, ryaw = self.ros.robot_pose
        
        # Robot velocity in world frame
        robot_vx = self.ros._last_odom_vx * math.cos(ryaw)
        robot_vy = self.ros._last_odom_vx * math.sin(ryaw)

        quintuplets = []
        for s in self.obs_slots:
            if not s["valid"]:
                quintuplets.append((0.0, 0.0, self.max_obstacle_depth, 0.0, 0.0))
                continue

            # Position in local frame
            lx, ly = _local_from_world(rx, ry, ryaw, float(s["wx"]), float(s["wy"]))
            d = math.hypot(lx, ly)
            d = max(0.0, min(d, self.max_obstacle_depth))
            
            # Relative velocity in world frame
            rel_vx_world = float(s["vx"]) - robot_vx
            rel_vy_world = float(s["vy"]) - robot_vy
            
            # Convert relative velocity to local frame
            rel_vx_local = math.cos(ryaw) * rel_vx_world + math.sin(ryaw) * rel_vy_world
            rel_vy_local = -math.sin(ryaw) * rel_vx_world + math.cos(ryaw) * rel_vy_world
            
            quintuplets.append((float(lx), float(ly), float(d), float(rel_vx_local), float(rel_vy_local)))

        quintuplets_sorted = sorted(quintuplets, key=lambda t: t[2])
        return quintuplets_sorted[:3]

    def _gt_goal_dist_bearing_rel_vel(self) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Returns (dist, bearing, rel_vx, rel_vy) where relative velocities are in robot's local frame"""
        if self.ros.robot_pose is None or self.ros.goal_pose_gt is None:
            return None, None, None, None
        
        rx, ry, ryaw = self.ros.robot_pose
        gx, gy = self.ros.goal_pose_gt
        
        # Distance and bearing
        dx = gx - rx
        dy = gy - ry
        dist = float(math.hypot(dx, dy))
        bearing = _wrap(math.atan2(dy, dx) - ryaw)
        
        # Relative velocity
        if self.ros.goal_vel_gt is not None:
            goal_vx, goal_vy = self.ros.goal_vel_gt
            
            # Robot velocity in world frame
            robot_vx = self.ros._last_odom_vx * math.cos(ryaw)
            robot_vy = self.ros._last_odom_vx * math.sin(ryaw)
            
            # Relative velocity in world frame
            rel_vx_world = goal_vx - robot_vx
            rel_vy_world = goal_vy - robot_vy
            
            # Convert to robot's local frame
            rel_vx_local = math.cos(ryaw) * rel_vx_world + math.sin(ryaw) * rel_vy_world
            rel_vy_local = -math.sin(ryaw) * rel_vx_world + math.cos(ryaw) * rel_vy_world
        else:
            rel_vx_local = 0.0
            rel_vy_local = 0.0
        
        return dist, bearing, float(rel_vx_local), float(rel_vy_local)

    # ---- FIX #4: Lateral-aware TTC ----
    def _compute_ttc(self, obs_quintuplets: List[Tuple[float, float, float, float, float]], v_forward: float) -> float:
        """
        Lateral-aware TTC estimate:
        - considers only obstacles in front (ox > 0)
        - filters out obstacles far to the side (|oy| > lateral margin)
        - uses forward distance ox (not Euclidean depth d)
        - considers relative velocity for more accurate TTC
        """
        if v_forward <= 1e-3:
            return float("inf")

        min_ttc = float("inf")
        for (ox, oy, d, rel_vx, rel_vy) in obs_quintuplets:
            if ox <= 0.0:
                continue
            # FIX #4: skip obstacles outside the scooter's projected path
            if abs(oy) > self.ttc_lateral_margin:
                continue
            
            # Relative approach speed (positive means approaching)
            approach_speed = v_forward - rel_vx
            
            if approach_speed > 1e-3:
                ttc = float(ox) / approach_speed
                if ttc < min_ttc:
                    min_ttc = ttc
        return min_ttc

    # ---- SAFETY SHIELD ----
    def _apply_safety_shield(
        self,
        v_cmd: float,
        w_cmd: float,
        ultra_fl: float,
        ultra_fr: float,
        ultra_ls: float,
        ultra_rs: float,
        obs_quintuplets: List[Tuple[float, float, float, float, float]],
    ) -> Tuple[float, float, str]:
        """
        Three-zone safety shield that overrides dangerous actions.

        Returns (safe_v, safe_w, shield_level) where shield_level is one of:
            "none"     — no intervention, action passes through
            "caution"  — forward speed capped
            "danger"   — strong deceleration + steering correction
            "critical" — full emergency stop

        The shield uses BOTH ultrasonic and camera-obstacle data, taking
        the most conservative (closest) threat from either source.

        Key design: reward penalties are still applied based on the ORIGINAL
        (pre-shield) action so the agent learns that the action was bad,
        even though the shield prevented the crash.
        """
        if not self.shield_enabled:
            return v_cmd, w_cmd, "none"

        safe_v = v_cmd
        safe_w = w_cmd
        level = "none"

        # --- Ultrasonic threat assessment ---
        min_front = min(ultra_fl, ultra_fr)
        min_side = min(ultra_ls, ultra_rs)
        min_ultra_all = min(min_front, min_side)

        # Determine which side is closer (for steering away)
        # Positive w = turn left, so if obstacle is on the left, steer right (negative w)
        ultra_left = min(ultra_fl, ultra_ls)
        ultra_right = min(ultra_fr, ultra_rs)

        # --- Camera obstacle threat assessment ---
        min_front_obs_depth = float("inf")
        closest_obs_oy = 0.0  # lateral offset of closest front obstacle
        for (ox, oy, d, rel_vx, rel_vy) in obs_quintuplets:
            if ox > 0.0 and abs(oy) < self.ttc_lateral_margin:
                if ox < min_front_obs_depth:
                    min_front_obs_depth = ox
                    closest_obs_oy = oy

        # --- Combine threats: pick the most urgent level ---

        # CRITICAL zone check
        if min_ultra_all < self.shield_critical_dist or min_front_obs_depth < self.shield_obs_critical:
            safe_v = 0.0
            safe_w = 0.0
            level = "critical"
            self._shield_interventions += 1
            return safe_v, safe_w, level

        # DANGER zone check
        in_danger = False
        if min_front < self.shield_danger_dist or min_front_obs_depth < self.shield_obs_danger:
            in_danger = True

        if in_danger:
            # Cap forward speed heavily
            if safe_v > self.shield_danger_max_v:
                safe_v = self.shield_danger_max_v

            # Steer away from the closest threat
            # Determine direction: use ultrasonics for close range, obs for medium range
            if min_front < self.shield_danger_dist:
                # Ultrasonic threat: steer away from closer side
                if ultra_left < ultra_right:
                    # Obstacle on left, steer right (negative w)
                    safe_w = safe_w - self.shield_danger_steer_bias
                else:
                    # Obstacle on right, steer left (positive w)
                    safe_w = safe_w + self.shield_danger_steer_bias
            elif min_front_obs_depth < self.shield_obs_danger:
                # Camera obstacle: steer away from its lateral position
                if closest_obs_oy >= 0:
                    # Obstacle on left, steer right
                    safe_w = safe_w - self.shield_danger_steer_bias
                else:
                    # Obstacle on right, steer left
                    safe_w = safe_w + self.shield_danger_steer_bias

            # Clamp angular velocity
            safe_w = max(-self.w_max, min(safe_w, self.w_max))

            level = "danger"
            self._shield_interventions += 1
            return safe_v, safe_w, level

        # CAUTION zone check
        in_caution = False
        if min_front < self.shield_caution_dist or min_front_obs_depth < self.shield_obs_caution:
            in_caution = True

        # Also check side ultrasonics for caution
        if min_side < self.shield_danger_dist:
            in_caution = True

        if in_caution:
            # Only limit forward speed, let policy handle steering
            if safe_v > self.shield_caution_max_v:
                safe_v = self.shield_caution_max_v
            level = "caution"
            self._shield_interventions += 1
            return safe_v, safe_w, level

        return safe_v, safe_w, level

    # --- Observation construction (normalized, with relative velocities) ---
    def _obs(self, default: bool = False):
        if default:
            return self._last_obs_valid.copy()

        if self.use_ground_truth_geometry:
            self._update_obstacle_memory_gt()

        dist, bearing, goal_rel_vx, goal_rel_vy = self._gt_goal_dist_bearing_rel_vel()
        if dist is None or bearing is None:
            return self._last_obs_valid.copy()

        ultra_vec, _ = self._ultra_vector()
        ultra_fl, ultra_fr, ultra_ls, ultra_rs = map(float, ultra_vec.tolist())

        obs_quintuplets = self._slots_local_quintuplets_sorted_by_threat()
        (x1, y1, d1, rvx1, rvy1), (x2, y2, d2, rvx2, rvy2), (x3, y3, d3, rvx3, rvy3) = obs_quintuplets

        v = float(self.ros._last_odom_vx)
        w = float(self.ros._last_odom_wz)

        # ---- Normalize all features to roughly [-1, 1] ----
        obs = np.array(
            [
                np.clip(dist / self.GOAL_DIST_MAX, 0.0, 1.0),
                bearing / math.pi,
                np.clip(goal_rel_vx / self.VEL_MAX, -1.0, 1.0),
                np.clip(goal_rel_vy / self.VEL_MAX, -1.0, 1.0),
                np.clip(v / self.v_max, -1.0, 1.0),
                np.clip(w / self.w_max, -1.0, 1.0),
                np.clip(x1 / self.OBS_COORD_MAX, -1.0, 1.0),
                np.clip(y1 / self.OBS_COORD_MAX, -1.0, 1.0),
                np.clip(d1 / self.max_obstacle_depth, 0.0, 1.0),
                np.clip(rvx1 / self.VEL_MAX, -1.0, 1.0),
                np.clip(rvy1 / self.VEL_MAX, -1.0, 1.0),
                np.clip(x2 / self.OBS_COORD_MAX, -1.0, 1.0),
                np.clip(y2 / self.OBS_COORD_MAX, -1.0, 1.0),
                np.clip(d2 / self.max_obstacle_depth, 0.0, 1.0),
                np.clip(rvx2 / self.VEL_MAX, -1.0, 1.0),
                np.clip(rvy2 / self.VEL_MAX, -1.0, 1.0),
                np.clip(x3 / self.OBS_COORD_MAX, -1.0, 1.0),
                np.clip(y3 / self.OBS_COORD_MAX, -1.0, 1.0),
                np.clip(d3 / self.max_obstacle_depth, 0.0, 1.0),
                np.clip(rvx3 / self.VEL_MAX, -1.0, 1.0),
                np.clip(rvy3 / self.VEL_MAX, -1.0, 1.0),
                ultra_fl / self.ultra_max_range,
                ultra_fr / self.ultra_max_range,
                ultra_ls / self.ultra_max_range,
                ultra_rs / self.ultra_max_range,
                # ---- previous action ----
                np.clip(self._prev_v_cmd / self.v_max, -1.0, 1.0),
                np.clip(self._prev_w_cmd / self.w_max, -1.0, 1.0),
            ],
            dtype=np.float32
        )

        if not np.all(np.isfinite(obs)):
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

        while accumulated < target_rotation:
            self.ros.send_cmd(0.0, w)
            self._spin(self.dt)

            if self.ros.goal_state is not None and getattr(self.ros.goal_state, "visible", False):
                self.ros.get_logger().info("[SEARCH] Goal seen during search.")
                break

            if self.ros.robot_pose is not None:
                _, _, yaw = self.ros.robot_pose
                dyaw = _wrap(yaw - prev_yaw)
                accumulated += abs(dyaw)
                prev_yaw = yaw

        self.ros.send_cmd(0.0, 0.0)
        self._spin(0.1)

        self.ros.get_logger().info(f"[SEARCH] Finished search; rot={accumulated:.2f} rad")
        return True

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
        self._prev_phi = 0.0

        self._ep_robot_traj = []
        self._robot_start = None
        self._goal_start = None
        self._robot_last = None
        self._goal_last = None
        self._min_dist = None
        self.prev_dist = None

        self._prev_v_cmd = 0.0
        self._prev_w_cmd = 0.0
        self._shield_interventions = 0
        self._shield_active = False

        for s in self.obs_slots:
            s["valid"] = False
            s["wx"] = 0.0
            s["wy"] = 0.0
            s["vx"] = 0.0
            s["vy"] = 0.0
            s["last_seen"] = 0.0

        self._call_service_cli("/pause_physics", "std_srvs/srv/Empty", "{}")
        self._hard_stop_robot(duration=0.5)

        self._reset_entity_with_retry("my_robot", 0.0, 0.0, 0.3, 0.7854)

        radius = 7.0
        angle = self.np_random.uniform(0.0, 2.0 * math.pi)
        gx = radius * math.cos(angle)
        gy = radius * math.sin(angle)
        self._reset_entity_with_retry("goal_marker", gx, gy, 0.75, 0.0)

        self._spin(0.3)

        subprocess.run(
            ["ros2", "service", "call", "/reset_random_obstacles", "std_srvs/srv/Trigger", "{}"],
            capture_output=True,
            text=True,
        )
        self.ros.get_logger().info("[ENV] Random obstacles reset/teleported")

        self._spin(0.3)

        self._call_service_cli("/unpause_physics", "std_srvs/srv/Empty", "{}")
        self._hard_stop_robot(duration=0.3)

        if self.enable_search_on_reset:
            _ = self._search_full_rotation()

        now = time.time()
        self._t0 = now

        if self.ros.robot_pose is not None:
            rx, ry, _ = self.ros.robot_pose
            self._robot_start = (rx, ry)
            self._robot_last = (rx, ry)
            self._ep_robot_traj.append((rx, ry))

        if self.ros.goal_pose is not None:
            gx0, gy0 = self.ros.goal_pose
            self._goal_start = (gx0, gy0)
            self._goal_last = (gx0, gy0)

        obs = self._obs(default=False)
        self._min_dist = float(obs[0]) * self.GOAL_DIST_MAX   # un-normalize for internal tracking

        self.ros.get_logger().info(
            f"[RESET] gamma={self.gamma:.3f} step_penalty={self.step_penalty:.3f} ttc_threshold={self.ttc_threshold:.2f} "
            f"smooth_alpha={self.smooth_alpha:.2f}"
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

        # Action smoothing (FIX #2: alpha raised to 0.70)
        v_cmd = (1.0 - self.smooth_alpha) * self._prev_v_cmd + self.smooth_alpha * v_des
        w_cmd = (1.0 - self.smooth_alpha) * self._prev_w_cmd + self.smooth_alpha * w_des

        # ---- FIX #2: compute action change BEFORE updating prev ----
        delta_v = abs(v_cmd - self._prev_v_cmd)
        delta_w = abs(w_cmd - self._prev_w_cmd)

        # Save the POLICY's intended command (pre-shield) for reward computation
        v_policy = v_cmd
        w_policy = w_cmd

        # ---- SAFETY SHIELD: read current sensor state and override if dangerous ----
        # We read ultrasonics and obstacle memory NOW (before sending cmd) to decide
        # whether to intervene. This uses data from the PREVIOUS spin cycle.
        pre_ultra_vec, _ = self._ultra_vector()
        pre_ultra_fl, pre_ultra_fr, pre_ultra_ls, pre_ultra_rs = map(float, pre_ultra_vec.tolist())

        if self.use_ground_truth_geometry:
            self._update_obstacle_memory_gt()
        pre_obs_quintuplets_raw = self._slots_local_quintuplets_sorted_by_threat()

        v_cmd, w_cmd, shield_level = self._apply_safety_shield(
            v_cmd, w_cmd,
            pre_ultra_fl, pre_ultra_fr, pre_ultra_ls, pre_ultra_rs,
            pre_obs_quintuplets_raw,
        )
        self._shield_active = (shield_level != "none")

        self._prev_v_cmd = v_cmd
        self._prev_w_cmd = w_cmd

        self.ros.send_cmd(v_cmd, w_cmd)
        self._spin(self.dt)

        obs = self._obs()
        if not np.all(np.isfinite(obs)):
            self.ros.get_logger().error(f"[STEP] Non-finite obs: {obs}, using last valid.")
            obs = self._last_obs_valid.copy()

        # Un-normalize goal distance for reward computation
        dist = float(obs[0]) * self.GOAL_DIST_MAX
        bearing = float(obs[1]) * math.pi   # un-normalize bearing

        v_obs = float(obs[4]) * self.v_max
        w_obs = float(obs[5]) * self.w_max

        # Un-normalize obstacle quintuplets for TTC / collision
        o1 = (
            float(obs[6]) * self.OBS_COORD_MAX, 
            float(obs[7]) * self.OBS_COORD_MAX, 
            float(obs[8]) * self.max_obstacle_depth,
            float(obs[9]) * self.VEL_MAX,
            float(obs[10]) * self.VEL_MAX
        )
        o2 = (
            float(obs[11]) * self.OBS_COORD_MAX, 
            float(obs[12]) * self.OBS_COORD_MAX, 
            float(obs[13]) * self.max_obstacle_depth,
            float(obs[14]) * self.VEL_MAX,
            float(obs[15]) * self.VEL_MAX
        )
        o3 = (
            float(obs[16]) * self.OBS_COORD_MAX, 
            float(obs[17]) * self.OBS_COORD_MAX, 
            float(obs[18]) * self.max_obstacle_depth,
            float(obs[19]) * self.VEL_MAX,
            float(obs[20]) * self.VEL_MAX
        )

        ultra_fl = float(obs[21]) * self.ultra_max_range
        ultra_fr = float(obs[22]) * self.ultra_max_range
        ultra_ls = float(obs[23]) * self.ultra_max_range
        ultra_rs = float(obs[24]) * self.ultra_max_range

        obs_quintuplets = [o1, o2, o3]
        min_obs_depth = float(min(o1[2], o2[2], o3[2]))
        min_ultra = float(min(ultra_fl, ultra_fr, ultra_ls, ultra_rs))

        # Logging trajectories
        if self.ros.robot_pose is not None:
            rx, ry, _ = self.ros.robot_pose
            self._robot_last = (rx, ry)
            self._ep_robot_traj.append((rx, ry))

        if self.ros.goal_pose is not None:
            gx, gy = self.ros.goal_pose
            self._goal_last = (gx, gy)

        if self._min_dist is None or dist < self._min_dist:
            self._min_dist = dist

        reward = 0.0
        terminated = False
        truncated = False
        reason = ""

        # (1) time penalty (FIX #1: reduced from 0.01 to 0.005)
        reward -= self.step_penalty

        # (2) TTC penalty ONLY when moving forward (FIX #4: lateral-aware)
        # NOTE: use the POLICY's intended speed for reward, not the shielded speed.
        # This way the agent learns "that action was dangerous" even if the shield saved it.
        ttc = self._compute_ttc(obs_quintuplets, max(0.0, v_policy))
        if v_policy > 0.1 and ttc < self.ttc_threshold:
            reward -= float(np.exp(-ttc))

        # (3) progress-based goal shaping (FIX #1: c_progress reduced to 2.0)
        if self.prev_dist is None:
            self.prev_dist = dist
        progress = self.prev_dist - dist
        reward += self.c_progress * progress
        self.prev_dist = dist

        # (3b) FIX #10-green: bearing alignment reward
        reward += self.c_bearing * math.cos(bearing)

        # (3c) FIX #2: action smoothness penalty (via reward, complementing higher alpha)
        reward -= self.c_action_change * (delta_v + delta_w)

        # (3d) FIX #9: ultrasonic proximity penalty zone
        if min_ultra < self.ultra_proximity_zone:
            reward -= self.c_ultra_proximity * (self.ultra_proximity_zone - min_ultra)

        # (3e) SHIELD PENALTY: penalize the agent when the shield had to intervene.
        # This teaches it to avoid situations that require shield activation.
        if shield_level == "critical":
            reward -= 5.0
        elif shield_level == "danger":
            reward -= 2.0
        elif shield_level == "caution":
            reward -= 0.5

        # (4) terminal events (FIX #1: much larger terminal rewards)
        if min_ultra < self.ultra_collision_radius or min_obs_depth < self.obstacle_collision_radius:
            reward += self.R_collision       # -50.0
            terminated = True
            reason = "collision"

        elif dist <= self.success_radius:
            reward += self.R_goal            # +50.0
            terminated = True
            reason = "goal"

        elif (time.time() - self._t0) >= self.time_limit:
            truncated = True
            reward -= 10.0                   # FIX #1: was -R_goal (= -1.0), now explicit -10
            reason = "timeout"


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
                {"valid": s["valid"], "wx": s["wx"], "wy": s["wy"], "vx": s["vx"], "vy": s["vy"], "last_seen": s["last_seen"]}
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

            "ttc": float(ttc),
            "v_cmd": float(v_cmd),
            "w_cmd": float(w_cmd),
            "v_policy": float(v_policy),
            "w_policy": float(w_policy),
            "shield_level": shield_level,
            "shield_interventions_ep": self._shield_interventions,
            "v_obs": float(v_obs),
            "w_obs": float(w_obs),
            "gamma_shaping": float(self.gamma),
            "step_penalty": float(self.step_penalty),
        }

        # Per-step debug payload for trajectory logging (kept)
        step_robot_pose = self.ros.robot_pose
        step_goal_pose = self.ros.goal_pose
        step_gt_obstacles = getattr(self.ros, "gt_obstacles_world", None)

        seen_local = self._parse_obstacles_local()
        seen_world = []
        if step_robot_pose is not None:
            rx, ry, ryaw = step_robot_pose
            for (lx, ly, d) in seen_local:
                wx, wy = _world_from_local(rx, ry, ryaw, lx, ly)
                seen_world.append((float(wx), float(wy), float(d)))

        info["step_robot_pose"] = step_robot_pose
        info["step_goal_pose"] = step_goal_pose
        info["step_goal_pose_gt"] = self.ros.goal_pose_gt
        info["step_goal_vel_gt"] = self.ros.goal_vel_gt
        info["step_gt_obstacles_world"] = step_gt_obstacles
        info["step_gt_obstacles_vel"] = self.ros.gt_obstacles_vel
        info["step_seen_obstacles_local"] = seen_local
        info["step_seen_obstacles_world"] = seen_world

        self.ros.get_logger().info(
            f"[STEP] dist={dist:.2f} bearing={bearing:.2f} v={v_cmd:.2f} w={w_cmd:.2f} "
            f"min_obs={min_obs_depth:.2f} min_ultra={min_ultra:.2f} ttc={ttc:.2f} "
            f"shield={shield_level} "
            f"R={reward:.3f} term={terminated} trunc={truncated} reason={reason}"
        )

        return obs, float(reward), terminated, truncated, info

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