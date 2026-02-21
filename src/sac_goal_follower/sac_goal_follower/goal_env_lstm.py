#!/usr/bin/env python3
"""
LSTM-Compatible Goal Following Environment

This environment extends goal_env_gt.py to support LSTM-based policies.
Main changes:
1. Returns dict observations instead of flat arrays
2. Separates obstacle sequence from fixed features
3. Handles variable number of obstacles (1-5)
"""

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
    """ROS2 wrapper - same as goal_env_gt.py"""

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
        super().__init__("sac_goal_env_lstm_node")

        self.robot_model_name = str(robot_model_name)
        self.goal_model_name = str(goal_model_name)

        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)

        self.goal_state: GoalMarkerState | None = None
        self.goal_pose = None
        self.robot_pose = None
        self._last_odom_vx = 0.0
        self._last_odom_wz = 0.0
        self.obstacles_json: str | None = None

        self.goal_pose_gt = None
        self.goal_vel_gt = None

        self.ultra = {
            "front_left": None,
            "front_right": None,
            "left_side": None,
            "right_side": None,
        }
        self.ultra_stamp = {k: 0.0 for k in self.ultra.keys()}

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

        if gt_obstacle_names is None:
            # 7 walking persons (indices 0-6) + 3 standing persons (indices 7-9)
            gt_obstacle_names = [
                "yolo_obstacle_0", "yolo_obstacle_1", "yolo_obstacle_2",
                "yolo_obstacle_3", "yolo_obstacle_4", "yolo_obstacle_5",
                "yolo_obstacle_6", "yolo_obstacle_7", "yolo_obstacle_8",
                "yolo_obstacle_9",
            ]
        self.gt_obstacle_names = [str(x) for x in gt_obstacle_names]
        self.gt_obstacles_world = {n: None for n in self.gt_obstacle_names}
        self.gt_obstacles_vel = {n: None for n in self.gt_obstacle_names}

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

        try:
            gi = msg.name.index(self.goal_model_name)
            pg = msg.pose[gi].position
            vg = msg.twist[gi].linear
            self.goal_pose_gt = (float(pg.x), float(pg.y))
            self.goal_vel_gt = (float(vg.x), float(vg.y))
        except ValueError:
            pass

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

    def send_cmd(self, v: float, w: float) -> None:
        tw = Twist()
        tw.linear.x = float(v)
        tw.angular.z = float(w)
        self.cmd_pub.publish(tw)

    def _ultra_cb(self, key: str, msg: Float32):
        self.ultra[key] = float(msg.data)
        self.ultra_stamp[key] = time.time()


class GoalFollowerLSTMEnv(Env):
    """
    LSTM-Compatible Goal Following Environment
    
    Returns dict observations with:
    - 'fixed': [goal_dist, goal_bearing, goal_rel_vx, goal_rel_vy, v, w, 
                ultra_fl, ultra_fr, ultra_ls, ultra_rs, prev_v, prev_w]  (12D)
    - 'obstacles': [[x, y, d, vx, vy], ...] for up to 5 obstacles (5×5)
    """
    
    metadata = {"render_modes": []}

    # Normalization constants
    GOAL_DIST_MAX = 15.0
    OBS_COORD_MAX = 10.0
    VEL_MAX = 2.0
    
    # Maximum number of obstacles
    MAX_OBSTACLES = 5

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
        R_goal=50.0,
        R_collision=-50.0,
        gamma_shaping=0.97,
        step_penalty=0.1,
        ttc_threshold=2.0,
        cam_x=0.4,
        cam_y=0.0,
        use_ground_truth_geometry: bool = True,
        goal_model_name: str = "goal_marker",
        gt_obstacle_names: Optional[List[str]] = None,
        smooth_alpha: float = 0.70,
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

        self.max_obstacle_depth = 10.0

        # Ultrasonic
        self.ultra_max_range = 1.0
        self.ultra_stale_sec = 0.5

        # --------- OBSTACLE MEMORY ----------
        self.num_obs_mem = self.MAX_OBSTACLES  # Now using 5 obstacles
        self.obs_match_dist = 1.2
        self.obs_expire_s = 30.0
        self.obs_slots: List[Dict[str, Any]] = [
            {"valid": False, "wx": 0.0, "wy": 0.0, "vx": 0.0, "vy": 0.0, "last_seen": 0.0} 
            for _ in range(self.num_obs_mem)
        ]

        # --------- OBSERVATION SPACE (Dict) ----------
        # Fixed features: 12D
        #   [goal_dist, goal_bearing, goal_rel_vx, goal_rel_vy,  # 4
        #    v, w,                                                 # 2
        #    ultra_fl, ultra_fr, ultra_ls, ultra_rs,              # 4
        #    prev_v, prev_w]                                       # 2
        # Obstacle sequence: 5×5 = 25D
        #   Each obstacle: [x, y, depth, rel_vx, rel_vy]
        
        self.observation_space = spaces.Dict({
            'fixed': spaces.Box(
                low=np.array([0.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32
            ),
            'obstacles': spaces.Box(
                low=np.array([[-1.0, -1.0, 0.0, -1.0, -1.0]] * self.MAX_OBSTACLES, dtype=np.float32),
                high=np.array([[1.0, 1.0, 1.0, 1.0, 1.0]] * self.MAX_OBSTACLES, dtype=np.float32),
                dtype=np.float32
            )
        })

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

        # Action smoothing
        self.smooth_alpha = float(np.clip(smooth_alpha, 0.0, 1.0))
        self._prev_v_cmd = 0.0
        self._prev_w_cmd = 0.0
        self.c_progress = 2.0

        # Reward coefficients
        self.c_action_change = 0.3
        self.ttc_lateral_margin = float(ttc_lateral_margin)
        self.ultra_proximity_zone = 0.5
        self.c_ultra_proximity = 2.0
        self.c_bearing = 0.3

        # Search-on-reset
        self.enable_search_on_reset = True
        self.search_angular_speed = 1.0

        # Collision thresholds
        self.obstacle_collision_radius = 0.7
        self.ultra_collision_radius = 0.2

        # --------- SAFETY SHIELD ----------
        self.shield_enabled = True
        self.shield_critical_dist = 0.25
        self.shield_danger_dist = 0.50
        self.shield_caution_dist = 0.80
        self.shield_obs_critical = 0.9
        self.shield_obs_danger = 1.5
        self.shield_obs_caution = 2.5
        self.shield_caution_max_v = 0.4
        self.shield_danger_max_v = 0.1
        self.shield_danger_steer_bias = 0.5
        self._shield_interventions = 0
        self._shield_active = False

        # ROS wrapper
        if gt_obstacle_names is None:
            # 7 walking persons (indices 0-6) + 3 standing persons (indices 7-9)
            gt_obstacle_names = [
                "yolo_obstacle_0", "yolo_obstacle_1", "yolo_obstacle_2",
                "yolo_obstacle_3", "yolo_obstacle_4", "yolo_obstacle_5",
                "yolo_obstacle_6", "yolo_obstacle_7", "yolo_obstacle_8",
                "yolo_obstacle_9",
            ]

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
        self.prev_dist = None

        # Default observation (valid dict structure)
        self._last_obs_valid = {
            'fixed': np.array([
                7.0 / self.GOAL_DIST_MAX, 0.0, 0.0, 0.0,  # goal
                0.0, 0.0,                                   # robot
                1.0, 1.0, 1.0, 1.0,                        # ultrasonics
                0.0, 0.0,                                   # prev action
            ], dtype=np.float32),
            'obstacles': np.array([
                [0.0, 0.0, 1.0, 0.0, 0.0],  # all obstacles set to far away
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
            ], dtype=np.float32)
        }

    def _spin(self, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            self.exec.spin_once(timeout_sec=0.001)

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
        
        robot_vx = self.ros._last_odom_vx * math.cos(ryaw)
        robot_vy = self.ros._last_odom_vx * math.sin(ryaw)

        quintuplets = []
        for s in self.obs_slots:
            if not s["valid"]:
                quintuplets.append((0.0, 0.0, self.max_obstacle_depth, 0.0, 0.0))
                continue

            lx, ly = _local_from_world(rx, ry, ryaw, float(s["wx"]), float(s["wy"]))
            d = math.hypot(lx, ly)
            d = max(0.0, min(d, self.max_obstacle_depth))
            
            rel_vx_world = float(s["vx"]) - robot_vx
            rel_vy_world = float(s["vy"]) - robot_vy
            
            rel_vx_local = math.cos(ryaw) * rel_vx_world + math.sin(ryaw) * rel_vy_world
            rel_vy_local = -math.sin(ryaw) * rel_vx_world + math.cos(ryaw) * rel_vy_world
            
            quintuplets.append((float(lx), float(ly), float(d), float(rel_vx_local), float(rel_vy_local)))

        quintuplets_sorted = sorted(quintuplets, key=lambda t: t[2])
        return quintuplets_sorted[:self.num_obs_mem]

    def _gt_goal_dist_bearing_rel_vel(self) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        if self.ros.robot_pose is None or self.ros.goal_pose_gt is None:
            return None, None, None, None
        
        rx, ry, ryaw = self.ros.robot_pose
        gx, gy = self.ros.goal_pose_gt
        
        dx = gx - rx
        dy = gy - ry
        dist = float(math.hypot(dx, dy))
        bearing = _wrap(math.atan2(dy, dx) - ryaw)
        
        if self.ros.goal_vel_gt is not None:
            goal_vx, goal_vy = self.ros.goal_vel_gt
            
            robot_vx = self.ros._last_odom_vx * math.cos(ryaw)
            robot_vy = self.ros._last_odom_vx * math.sin(ryaw)
            
            rel_vx_world = goal_vx - robot_vx
            rel_vy_world = goal_vy - robot_vy
            
            rel_vx_local = math.cos(ryaw) * rel_vx_world + math.sin(ryaw) * rel_vy_world
            rel_vy_local = -math.sin(ryaw) * rel_vx_world + math.cos(ryaw) * rel_vy_world
        else:
            rel_vx_local = 0.0
            rel_vy_local = 0.0
        
        return dist, bearing, float(rel_vx_local), float(rel_vy_local)

    def _compute_ttc(self, obs_quintuplets: List[Tuple[float, float, float, float, float]], v_forward: float) -> float:
        if v_forward <= 1e-3:
            return float("inf")

        min_ttc = float("inf")
        for (ox, oy, d, rel_vx, rel_vy) in obs_quintuplets:
            if ox <= 0.0:
                continue
            if abs(oy) > self.ttc_lateral_margin:
                continue
            
            approach_speed = v_forward - rel_vx
            
            if approach_speed > 1e-3:
                ttc = float(ox) / approach_speed
                if ttc < min_ttc:
                    min_ttc = ttc
        return min_ttc

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
        if not self.shield_enabled:
            return v_cmd, w_cmd, "none"

        safe_v = v_cmd
        safe_w = w_cmd
        level = "none"

        min_front = min(ultra_fl, ultra_fr)
        min_side = min(ultra_ls, ultra_rs)
        min_ultra_all = min(min_front, min_side)

        ultra_left = min(ultra_fl, ultra_ls)
        ultra_right = min(ultra_fr, ultra_rs)

        min_front_obs_depth = float("inf")
        closest_obs_oy = 0.0
        for (ox, oy, d, rel_vx, rel_vy) in obs_quintuplets:
            if ox > 0.0 and abs(oy) < self.ttc_lateral_margin:
                if ox < min_front_obs_depth:
                    min_front_obs_depth = ox
                    closest_obs_oy = oy

        if min_ultra_all < self.shield_critical_dist or min_front_obs_depth < self.shield_obs_critical:
            safe_v = 0.0
            safe_w = 0.0
            level = "critical"
            self._shield_interventions += 1
            return safe_v, safe_w, level

        in_danger = False
        if min_front < self.shield_danger_dist or min_front_obs_depth < self.shield_obs_danger:
            in_danger = True

        if in_danger:
            if safe_v > self.shield_danger_max_v:
                safe_v = self.shield_danger_max_v

            if min_front < self.shield_danger_dist:
                if ultra_left < ultra_right:
                    safe_w = safe_w - self.shield_danger_steer_bias
                else:
                    safe_w = safe_w + self.shield_danger_steer_bias
            elif min_front_obs_depth < self.shield_obs_danger:
                if closest_obs_oy >= 0:
                    safe_w = safe_w - self.shield_danger_steer_bias
                else:
                    safe_w = safe_w + self.shield_danger_steer_bias

            safe_w = max(-self.w_max, min(safe_w, self.w_max))

            level = "danger"
            self._shield_interventions += 1
            return safe_v, safe_w, level

        in_caution = False
        if min_front < self.shield_caution_dist or min_front_obs_depth < self.shield_obs_caution:
            in_caution = True

        if min_side < self.shield_danger_dist:
            in_caution = True

        if in_caution:
            if safe_v > self.shield_caution_max_v:
                safe_v = self.shield_caution_max_v
            level = "caution"
            self._shield_interventions += 1
            return safe_v, safe_w, level

        return safe_v, safe_w, level

    def _obs(self, default: bool = False) -> Dict[str, np.ndarray]:
        """
        Returns dict observation:
        {
            'fixed': [12D] - goal, robot, ultrasonics, prev_action
            'obstacles': [6×5] - obstacle sequence (x, y, d, vx, vy)
        }
        """
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

        v = float(self.ros._last_odom_vx)
        w = float(self.ros._last_odom_wz)

        # Build fixed features (12D)
        fixed = np.array([
            np.clip(dist / self.GOAL_DIST_MAX, 0.0, 1.0),
            bearing / math.pi,
            np.clip(goal_rel_vx / self.VEL_MAX, -1.0, 1.0),
            np.clip(goal_rel_vy / self.VEL_MAX, -1.0, 1.0),
            np.clip(v / self.v_max, -1.0, 1.0),
            np.clip(w / self.w_max, -1.0, 1.0),
            ultra_fl / self.ultra_max_range,
            ultra_fr / self.ultra_max_range,
            ultra_ls / self.ultra_max_range,
            ultra_rs / self.ultra_max_range,
            np.clip(self._prev_v_cmd / self.v_max, -1.0, 1.0),
            np.clip(self._prev_w_cmd / self.w_max, -1.0, 1.0),
        ], dtype=np.float32)

        # Build obstacle sequence (5×5)
        obstacles = np.zeros((self.MAX_OBSTACLES, 5), dtype=np.float32)
        for i, (x, y, d, rvx, rvy) in enumerate(obs_quintuplets):
            obstacles[i, 0] = np.clip(x / self.OBS_COORD_MAX, -1.0, 1.0)
            obstacles[i, 1] = np.clip(y / self.OBS_COORD_MAX, -1.0, 1.0)
            obstacles[i, 2] = np.clip(d / self.max_obstacle_depth, 0.0, 1.0)
            obstacles[i, 3] = np.clip(rvx / self.VEL_MAX, -1.0, 1.0)
            obstacles[i, 4] = np.clip(rvy / self.VEL_MAX, -1.0, 1.0)

        obs_dict = {
            'fixed': fixed,
            'obstacles': obstacles
        }

        # Validate
        if not (np.all(np.isfinite(fixed)) and np.all(np.isfinite(obstacles))):
            return self._last_obs_valid.copy()

        self._last_obs_valid = obs_dict
        return obs_dict

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
        self._min_dist = float(obs['fixed'][0]) * self.GOAL_DIST_MAX

        self.ros.get_logger().info(
            f"[RESET] gamma={self.gamma:.3f} step_penalty={self.step_penalty:.3f} ttc_threshold={self.ttc_threshold:.2f} "
            f"smooth_alpha={self.smooth_alpha:.2f}"
        )
        return obs, {}

    def step(self, action):
        a = np.clip(action, self.action_low, self.action_high).astype(np.float32)
        if not np.all(np.isfinite(a)):
            self.ros.get_logger().error(f"[STEP] Non-finite action: {a}, zeroing.")
            a = np.array([0.0, 0.0], dtype=np.float32)

        v_des = float(a[0]) * self.v_max
        w_des = float(a[1]) * self.w_max

        v_cmd = (1.0 - self.smooth_alpha) * self._prev_v_cmd + self.smooth_alpha * v_des
        w_cmd = (1.0 - self.smooth_alpha) * self._prev_w_cmd + self.smooth_alpha * w_des

        delta_v = abs(v_cmd - self._prev_v_cmd)
        delta_w = abs(w_cmd - self._prev_w_cmd)

        v_policy = v_cmd
        w_policy = w_cmd

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
        if not (np.all(np.isfinite(obs['fixed'])) and np.all(np.isfinite(obs['obstacles']))):
            self.ros.get_logger().error(f"[STEP] Non-finite obs, using last valid.")
            obs = self._last_obs_valid.copy()

        # Un-normalize for reward computation
        dist = float(obs['fixed'][0]) * self.GOAL_DIST_MAX
        bearing = float(obs['fixed'][1]) * math.pi

        v_obs = float(obs['fixed'][4]) * self.v_max
        w_obs = float(obs['fixed'][5]) * self.w_max

        # Un-normalize obstacles
        obs_quintuplets = []
        for i in range(self.MAX_OBSTACLES):
            ox = float(obs['obstacles'][i, 0]) * self.OBS_COORD_MAX
            oy = float(obs['obstacles'][i, 1]) * self.OBS_COORD_MAX
            od = float(obs['obstacles'][i, 2]) * self.max_obstacle_depth
            ovx = float(obs['obstacles'][i, 3]) * self.VEL_MAX
            ovy = float(obs['obstacles'][i, 4]) * self.VEL_MAX
            obs_quintuplets.append((ox, oy, od, ovx, ovy))

        min_obs_depth = float(min(o[2] for o in obs_quintuplets))

        ultra_fl = float(obs['fixed'][6]) * self.ultra_max_range
        ultra_fr = float(obs['fixed'][7]) * self.ultra_max_range
        ultra_ls = float(obs['fixed'][8]) * self.ultra_max_range
        ultra_rs = float(obs['fixed'][9]) * self.ultra_max_range
        min_ultra = float(min(ultra_fl, ultra_fr, ultra_ls, ultra_rs))

        # Logging
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

        reward -= self.step_penalty

        ttc = self._compute_ttc(obs_quintuplets, max(0.0, v_policy))
        if v_policy > 0.1 and ttc < self.ttc_threshold:
            reward -= float(np.exp(-ttc))

        if self.prev_dist is None:
            self.prev_dist = dist
        progress = self.prev_dist - dist
        reward += self.c_progress * progress
        self.prev_dist = dist

        reward += self.c_bearing * math.cos(bearing)

        reward -= self.c_action_change * (delta_v + delta_w)

        if min_ultra < self.ultra_proximity_zone:
            reward -= self.c_ultra_proximity * (self.ultra_proximity_zone - min_ultra)

        if shield_level == "critical":
            reward -= 5.0
        elif shield_level == "danger":
            reward -= 2.0
        elif shield_level == "caution":
            reward -= 0.5

        if min_ultra < self.ultra_collision_radius or min_obs_depth < self.obstacle_collision_radius:
            reward += self.R_collision
            terminated = True
            reason = "collision"

        elif dist <= self.success_radius:
            reward += self.R_goal
            terminated = True
            reason = "goal"

        elif (time.time() - self._t0) >= self.time_limit:
            truncated = True
            reward -= 10.0
            reason = "timeout"

        info = {
            "reason": reason,
            "robot_start": self._robot_start,
            "goal_start": self._goal_start,
            "robot_final": self._robot_last,
            "goal_final": self._goal_last,
            "min_dist": self._min_dist,
            "robot_traj": self._ep_robot_traj,
            "obstacles_local_sorted": obs_quintuplets[:3],  # For backward compatibility
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
