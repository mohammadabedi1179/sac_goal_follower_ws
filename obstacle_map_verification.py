#!/usr/bin/env python3
import json
import math
import time
import itertools
from typing import Optional, Tuple, Dict, List, Any

import numpy as np
import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import String

import matplotlib.pyplot as plt

from detectors_msgs.msg import GoalMarkerState


def quat_to_yaw(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def _wrap(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def world_from_local(rx: float, ry: float, ryaw: float, lx: float, ly: float) -> Tuple[float, float]:
    # base_link local (x forward, y left) -> world
    xw = rx + lx * math.cos(ryaw) - ly * math.sin(ryaw)
    yw = ry + lx * math.sin(ryaw) + ly * math.cos(ryaw)
    return float(xw), float(yw)


def _solve_assignment_hungarian(cost: List[List[float]]) -> List[Tuple[int, int]]:
    """
    Min-cost assignment using Hungarian algorithm via SciPy if available.
    Returns list of (row_i, col_j) for row count = n.
    Handles rectangular matrices.
    """
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore
    except Exception:
        return []

    if not cost or not cost[0]:
        return []

    c = np.array(cost, dtype=np.float64)
    # Replace inf/nan with a large number so solver still works
    bad = ~np.isfinite(c)
    if np.any(bad):
        c[bad] = 1e9

    row_ind, col_ind = linear_sum_assignment(c)
    return list(zip(row_ind.tolist(), col_ind.tolist()))


def _solve_assignment_greedy(cost: List[List[float]]) -> List[Tuple[int, int]]:
    """
    SciPy-free fallback: greedy one-to-one matching by smallest cost.
    Works well for live plots and avoids factorial blowups for n=10.
    """
    if not cost or not cost[0]:
        return []
    n = len(cost)
    m = len(cost[0])

    triples: List[Tuple[float, int, int]] = []
    for i in range(n):
        for j in range(m):
            c = cost[i][j]
            if math.isfinite(c):
                triples.append((c, i, j))

    triples.sort(key=lambda t: t[0])
    used_rows = set()
    used_cols = set()
    pairs: List[Tuple[int, int]] = []
    for c, i, j in triples:
        if i in used_rows or j in used_cols:
            continue
        used_rows.add(i)
        used_cols.add(j)
        pairs.append((i, j))
        if len(pairs) >= min(n, m):
            break
    return pairs


def _solve_assignment_min_cost(cost: List[List[float]]) -> List[Tuple[int, int]]:
    """
    Wrapper:
      1) Try Hungarian (SciPy) if available.
      2) Else greedy one-to-one fallback (safe for n=10).
    """
    pairs = _solve_assignment_hungarian(cost)
    if pairs:
        return pairs
    return _solve_assignment_greedy(cost)


class ObstacleMapDebugger(Node):
    """
    Debugger node:
      - Reads obstacle detections JSON containing x_m,y_m in base_link.
      - Reads Gazebo /model_states for ground truth of robot + obstacles + goal marker.
      - Converts local (base_link) obstacles -> world using robot yaw.
      - Matches detections to GT obstacles via one-to-one assignment.
      - Reads GoalMarkerState and estimates goal world pose exactly like the RL env.
      - Plots live global map + error vectors + robot path.

    Conventions:
      - base_link: +x forward, +y left
      - goal bearing: camera RIGHT positive -> convert to yaw LEFT positive via b_yaw = -b_cam
    """

    def __init__(self):
        super().__init__("obstacle_map_debugger")

        # --- Names / Topics ---
        self.robot_name = "my_robot"

        # Up to 10 GT obstacles
        self.max_gt_obstacles = 10
        self.gt_obstacle_names = [f"yolo_obstacle_{i}" for i in range(self.max_gt_obstacles)]

        # Moving vs Static obstacle indices
        self.moving_ids = set(range(0, 7))   # 0..6 moving
        self.static_ids = set(range(7, 10))  # 7..9 static

        self.goal_name = "goal_marker"  # Gazebo model name

        self.obstacles_topic = "/follower_robot/obstacles_depth"
        self.goal_state_topic = "/follower_robot/depth_cam/goal_marker_state"
        self.model_states_topic = "/model_states"

        # Camera offset (MUST match your env)
        self.cam_x = 0.4
        self.cam_y = 0.0

        # Plot scale
        self.grid_scale = 13.0

        # Timeout for keeping last estimate alive
        self.detection_timeout_s = 0.1

        # Ignore obstacle detections with too large depth
        self.max_depth_accept = 50.0

        # Matching gate for obstacles (meters)
        self.match_gate_m = 3.0

        # ---------------- Path (episode trajectory) ----------------
        # Episode resets typically "teleport" the robot. We detect that and clear the path.
        self.path: List[Tuple[float, float]] = []
        self.last_robot_xy: Optional[Tuple[float, float]] = None
        self.last_robot_t: float = 0.0

        self.path_min_step_m = 0.03         # don’t add points if movement is tiny
        self.path_reset_jump_m = 2.0        # if robot jumps more than this between updates, treat as episode reset
        self.path_reset_time_gap_s = 1.0    # if updates pause (sim reset), treat as episode reset
        self.path_max_points = 4000         # keep memory bounded

        # --- State buffers ---
        self.robot_pose: Optional[Tuple[float, float, float]] = None  # (x,y,yaw)

        # Ground truth: obstacles + goal
        self.gt_world: Dict[str, Optional[Tuple[float, float]]] = {n: None for n in self.gt_obstacle_names}
        self.gt_goal_world: Optional[Tuple[float, float]] = None

        # Latest estimated world positions (matched to GT obstacles)
        self.est_world: Dict[str, Optional[Tuple[float, float]]] = {n: None for n in self.gt_obstacle_names}
        self.est_local: Dict[str, Optional[Tuple[float, float, float]]] = {n: None for n in self.gt_obstacle_names}
        self.est_stamp: Dict[str, float] = {n: 0.0 for n in self.gt_obstacle_names}

        # Goal estimate (env-style)
        self.goal_state: Optional[GoalMarkerState] = None
        self.last_goal_pose_est_world: Optional[Tuple[float, float]] = None  # same as env's _last_goal_pose
        self.est_goal_world: Optional[Tuple[float, float]] = None
        self.est_goal_stamp: float = 0.0
        self.est_goal_dbg: Dict[str, float] = {}  # store depth/bearing/etc for text

        # --- ROS subs ---
        self.create_subscription(ModelStates, self.model_states_topic, self._model_states_cb, 10)
        self.create_subscription(String, self.obstacles_topic, self._obstacles_cb, 10)
        self.create_subscription(GoalMarkerState, self.goal_state_topic, self._goal_state_cb, 10)

        # ---------------- Plot theme ----------------
        self._apply_fancy_style()

        # --- Matplotlib live figure ---
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8.5, 7.5))
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlim(-self.grid_scale, self.grid_scale)
        self.ax.set_ylim(-self.grid_scale, self.grid_scale)
        self.ax.grid(True, alpha=0.25)
        self.ax.set_title("Map: obstacles + goal (estimated vs ground truth)", pad=12)
        self.ax.set_xlabel("X world [m]")
        self.ax.set_ylabel("Y world [m]")

        # Robot artists
        self.robot_point, = self.ax.plot([], [], marker="o", linestyle="", label="robot")
        self.robot_heading, = self.ax.plot([], [], linewidth=2.2, label="robot heading")

        # Robot path (two-layer line for a subtle “glow”)
        self.robot_path_glow, = self.ax.plot([], [], linewidth=6.0, alpha=0.12, label="_path_glow")
        self.robot_path, = self.ax.plot([], [], linewidth=2.2, alpha=0.9, label="robot path")

        # Obstacles artists (create 10 sets)
        self.est_points: Dict[str, Any] = {}
        self.gt_points: Dict[str, Any] = {}
        self.err_lines: Dict[str, Any] = {}

        # Markers: moving vs static
        moving_est_markers = ["x", "*", "s", "D", "P", "v", "X"]
        moving_gt_markers  = ["+", "^", "o", ">", "<", "h", "p"]

        static_est_marker = "1"   # distinctive tick marker
        static_gt_marker  = "8"   # octagon-like marker

        for name in self.gt_obstacle_names:
            idx = int(name.split("_")[-1])

            if idx in self.static_ids:
                est_m = static_est_marker
                gt_m = static_gt_marker
                est_ms = 12
                gt_ms = 11
                est_lbl = f"est {name} (static)"
                gt_lbl = f"gt  {name} (static)"
            else:
                est_m = moving_est_markers[idx % len(moving_est_markers)]
                gt_m = moving_gt_markers[idx % len(moving_gt_markers)]
                est_ms = 10
                gt_ms = 10
                est_lbl = f"est {name}"
                gt_lbl = f"gt  {name}"

            est_artist, = self.ax.plot([], [], marker=est_m, linestyle="", markersize=est_ms, label=est_lbl)
            gt_artist,  = self.ax.plot([], [], marker=gt_m,  linestyle="", markersize=gt_ms,  label=gt_lbl)
            err_artist, = self.ax.plot([], [], linestyle="--", linewidth=1.0, alpha=0.75)

            self.est_points[name] = est_artist
            self.gt_points[name] = gt_artist
            self.err_lines[name] = err_artist

        # Goal artists
        self.goal_gt_point, = self.ax.plot([], [], marker="^", linestyle="", markersize=12, label="gt goal")
        self.goal_est_point, = self.ax.plot([], [], marker="x", linestyle="", markersize=12, label="est goal (env-style)")
        self.goal_err_line, = self.ax.plot([], [], linestyle="--", linewidth=1.2, alpha=0.9)

        self.text = self.ax.text(
            0.02, 0.98, "", transform=self.ax.transAxes,
            ha="left", va="top", fontsize=9.5,
            bbox=dict(facecolor="black" if self._dark_mode else "white",
                      alpha=0.60 if self._dark_mode else 0.85,
                      edgecolor="none", boxstyle="round,pad=0.35")
        )

        # Matplotlib compatibility: older versions use ncol (not ncols)
        self.ax.legend(loc="lower right", ncol=2, fontsize=8.5, framealpha=0.6)

        # Update timer
        self.create_timer(0.1, self._update_plot)

        self.get_logger().info(
            "ObstacleMapDebugger started.\n"
            f"  obstacles_topic:   {self.obstacles_topic}\n"
            f"  goal_state_topic:  {self.goal_state_topic}\n"
            f"  model_states:      {self.model_states_topic}\n"
            f"  robot_name:        {self.robot_name}\n"
            f"  gt obstacles:      {self.gt_obstacle_names}\n"
            f"  goal_name:         {self.goal_name}\n"
            f"  cam_offset:        (cam_x={self.cam_x}, cam_y={self.cam_y})\n"
            f"  match_gate_m:      {self.match_gate_m} m\n"
            f"  grid_scale:        +/-{self.grid_scale} m\n"
            f"  path reset:        jump>{self.path_reset_jump_m}m or gap>{self.path_reset_time_gap_s}s\n"
            f"  moving IDs:        {sorted(self.moving_ids)}\n"
            f"  static IDs:        {sorted(self.static_ids)}"
        )

    def _apply_fancy_style(self):
        """
        Try a few nicer matplotlib styles. If not available, fall back gracefully.
        """
        styles_to_try = [
            "seaborn-v0_8-darkgrid",
            "seaborn-darkgrid",
            "dark_background",
            "ggplot",
        ]
        applied = False
        for s in styles_to_try:
            try:
                plt.style.use(s)
                applied = True
                break
            except Exception:
                continue

        # detect whether we’re effectively in a dark theme
        self._dark_mode = False
        if applied:
            fc = plt.rcParams.get("axes.facecolor", "white")
            if isinstance(fc, str) and fc.lower() in ["black", "#000000", "0.0"]:
                self._dark_mode = True

    # ---------------- callbacks ----------------

    def _goal_state_cb(self, msg: GoalMarkerState):
        self.goal_state = msg

    def _model_states_cb(self, msg: ModelStates):
        # robot pose
        try:
            i = msg.name.index(self.robot_name)
            p = msg.pose[i].position
            q = msg.pose[i].orientation
            yaw = quat_to_yaw(q)
            self.robot_pose = (float(p.x), float(p.y), float(yaw))
        except ValueError:
            pass

        # ground truth obstacle poses (up to 10)
        for n in self.gt_obstacle_names:
            try:
                j = msg.name.index(n)
                p2 = msg.pose[j].position
                self.gt_world[n] = (float(p2.x), float(p2.y))
            except ValueError:
                pass

        # ground truth goal pose
        try:
            g = msg.name.index(self.goal_name)
            pg = msg.pose[g].position
            self.gt_goal_world = (float(pg.x), float(pg.y))
        except ValueError:
            pass

    def _obstacles_cb(self, msg: String):
        """
        Incoming JSON example (list):
        [{"id":"9936","class":"person","depth_m":...,"bearing_rad":...,"x_m":...,"y_m":...}, ...]
        """
        if self.robot_pose is None:
            return

        try:
            objs = json.loads(msg.data)
        except Exception:
            return
        if not isinstance(objs, list) or len(objs) == 0:
            return

        rx, ry, ryaw = self.robot_pose
        now = time.time()

        # Build obstacle candidates in WORLD
        obstacle_candidates: List[Dict[str, Any]] = []
        for o in objs:
            try:
                lx = float(o.get("x_m", float("nan")))
                ly = float(o.get("y_m", float("nan")))
                d  = float(o.get("depth_m", float("nan")))
                b  = float(o.get("bearing_rad", float("nan")))

                if not (math.isfinite(lx) and math.isfinite(ly) and math.isfinite(d)):
                    continue
                if d <= 0.0 or d > self.max_depth_accept:
                    continue

                wx, wy = world_from_local(rx, ry, ryaw, lx, ly)
                obstacle_candidates.append({"wx": wx, "wy": wy, "lx": lx, "ly": ly, "d": d, "bearing": b})
            except Exception:
                continue

        if not obstacle_candidates:
            return

        # Collect available GT obstacles
        gt_list: List[Tuple[float, float]] = []
        gt_names: List[str] = []
        for name in self.gt_obstacle_names:
            gt = self.gt_world.get(name, None)
            if gt is not None:
                gt_names.append(name)
                gt_list.append(gt)

        if not gt_list:
            return

        n = len(gt_list)
        m = len(obstacle_candidates)

        matched: Dict[str, Dict[str, Any]] = {}

        # Build cost matrix
        cost = []
        for (gx, gy) in gt_list:
            cost.append([math.hypot(gx - c["wx"], gy - c["wy"]) for c in obstacle_candidates])

        pairs = _solve_assignment_min_cost(cost)
        for (i, j) in pairs:
            dist = cost[i][j]
            if dist <= self.match_gate_m:
                matched[gt_names[i]] = obstacle_candidates[j]

        for name, c in matched.items():
            self.est_world[name] = (float(c["wx"]), float(c["wy"]))
            self.est_local[name] = (float(c["lx"]), float(c["ly"]), float(c["d"]))
            self.est_stamp[name] = now

    # ---------------- goal estimation (env-style) ----------------

    def _update_goal_estimate_env_style(self):
        """
        Mirror GoalFollowerEnv._obs() goal estimation:
        - Use GoalMarkerState (depth + bearing).
        - bearing: camera RIGHT positive -> yaw LEFT positive => b_yaw = -b_cam
        - camera origin offset (cam_x, cam_y) from base_link.
        - world raycast from camera.
        """
        if self.robot_pose is None:
            return

        st = self.goal_state
        if st is None or (not bool(getattr(st, "visible", False))):
            return

        depth_cam = float(getattr(st, "depth_m", float("nan")))
        b_cam = float(getattr(st, "bearing_rad", float("nan")))

        if not (math.isfinite(depth_cam) and depth_cam > 0.0 and math.isfinite(b_cam)):
            return

        rx, ry, ryaw = self.robot_pose

        # camera origin in world (same as env)
        rx_cam = rx + self.cam_x * math.cos(ryaw) - self.cam_y * math.sin(ryaw)
        ry_cam = ry + self.cam_x * math.sin(ryaw) + self.cam_y * math.cos(ryaw)

        # bearing sign fix (same as env)
        b_yaw = -b_cam
        theta = ryaw + b_yaw

        gx = rx_cam + depth_cam * math.cos(theta)
        gy = ry_cam + depth_cam * math.sin(theta)

        self.last_goal_pose_est_world = (float(gx), float(gy))
        self.est_goal_world = (float(gx), float(gy))
        self.est_goal_stamp = time.time()
        self.est_goal_dbg = {
            "depth_m": depth_cam,
            "bearing_cam_rad": b_cam,
            "bearing_yaw_rad": b_yaw,
            "bearing_wrapped": _wrap(b_yaw),
        }

    # ---------------- path management ----------------

    def _update_robot_path(self, now: float):
        if self.robot_pose is None:
            return

        rx, ry, _ = self.robot_pose
        cur = (rx, ry)

        # First point
        if self.last_robot_xy is None:
            self.path = [cur]
            self.last_robot_xy = cur
            self.last_robot_t = now
            return

        # Reset conditions: big jump (teleport) or time gap
        dt = now - self.last_robot_t
        jump = math.hypot(cur[0] - self.last_robot_xy[0], cur[1] - self.last_robot_xy[1])

        if dt > self.path_reset_time_gap_s or jump > self.path_reset_jump_m:
            self.path = [cur]
            self.last_robot_xy = cur
            self.last_robot_t = now
            return

        # Append only if moved enough
        if jump >= self.path_min_step_m:
            self.path.append(cur)
            if len(self.path) > self.path_max_points:
                self.path = self.path[-self.path_max_points :]
            self.last_robot_xy = cur
            self.last_robot_t = now

    # ---------------- plotting ----------------

    def _update_plot(self):
        self.ax.set_xlim(-self.grid_scale, self.grid_scale)
        self.ax.set_ylim(-self.grid_scale, self.grid_scale)

        now = time.time()

        # Update robot path first
        self._update_robot_path(now)

        # Robot point + heading
        if self.robot_pose is not None:
            rx, ry, ryaw = self.robot_pose
            self.robot_point.set_data([rx], [ry])

            hx = rx + 0.8 * math.cos(ryaw)
            hy = ry + 0.8 * math.sin(ryaw)
            self.robot_heading.set_data([rx, hx], [ry, hy])
        else:
            self.robot_point.set_data([], [])
            self.robot_heading.set_data([], [])

        # Robot path line
        if len(self.path) >= 2:
            xs = [p[0] for p in self.path]
            ys = [p[1] for p in self.path]
            self.robot_path_glow.set_data(xs, ys)
            self.robot_path.set_data(xs, ys)
        else:
            self.robot_path_glow.set_data([], [])
            self.robot_path.set_data([], [])

        # Update goal estimate from GoalMarkerState (env-style)
        self._update_goal_estimate_env_style()

        lines = []
        errs = []

        # Obstacles
        for name in self.gt_obstacle_names:
            gt = self.gt_world.get(name, None)
            if gt is not None:
                gx, gy = gt
                self.gt_points[name].set_data([gx], [gy])
            else:
                self.gt_points[name].set_data([], [])

            est = self.est_world.get(name, None)
            fresh = est is not None and (now - float(self.est_stamp.get(name, 0.0))) <= self.detection_timeout_s

            if not fresh:
                self.est_points[name].set_data([], [])
                self.err_lines[name].set_data([], [])
                continue

            ex, ey = est
            self.est_points[name].set_data([ex], [ey])

            if gt is not None:
                gx, gy = gt
                self.err_lines[name].set_data([ex, gx], [ey, gy])
                e = math.hypot(gx - ex, gy - ey)
                errs.append(e)

                loc = self.est_local.get(name, None)
                if loc is not None:
                    lx, ly, d = loc
                    side = "CENTER-ish"
                    if ly < -1e-6:
                        side = "RIGHT (y_m<0, bearing>0 expected)"
                    elif ly > 1e-6:
                        side = "LEFT  (y_m>0, bearing<0 expected)"
                    lines.append(f"{name}: err={e:.3f} m | local x={lx:.2f} y={ly:.2f} d={d:.2f} | {side}")
                else:
                    lines.append(f"{name}: err={e:.3f} m")
            else:
                self.err_lines[name].set_data([], [])
                lines.append(f"{name}: GT not available yet")

        # Goal plotting
        goal_lines = []
        goal_err = None

        # GT goal
        if self.gt_goal_world is not None:
            gx, gy = self.gt_goal_world
            self.goal_gt_point.set_data([gx], [gy])
        else:
            self.goal_gt_point.set_data([], [])

        # EST goal (fresh)
        goal_fresh = self.est_goal_world is not None and (now - float(self.est_goal_stamp)) <= self.detection_timeout_s
        if goal_fresh:
            ex, ey = self.est_goal_world
            self.goal_est_point.set_data([ex], [ey])
        else:
            self.goal_est_point.set_data([], [])
            self.goal_err_line.set_data([], [])

        if goal_fresh and self.gt_goal_world is not None:
            gx, gy = self.gt_goal_world
            ex, ey = self.est_goal_world
            self.goal_err_line.set_data([ex, gx], [ey, gy])
            goal_err = math.hypot(gx - ex, gy - ey)

            dbg = self.est_goal_dbg or {}
            goal_lines.append(
                "goal(env): "
                f"err={goal_err:.3f} m | depth={dbg.get('depth_m', float('nan')):.2f} "
                f"| b_cam={dbg.get('bearing_cam_rad', float('nan')):.2f} "
                f"| b_yaw={dbg.get('bearing_yaw_rad', float('nan')):.2f}"
            )
        elif self.gt_goal_world is None:
            goal_lines.append("goal: GT not available yet")
        elif not goal_fresh:
            goal_lines.append("goal: waiting for fresh estimate...")

        mean_err = float(np.mean(errs)) if errs else float("nan")
        header = (
            f"Obstacles (fresh<= {self.detection_timeout_s:.1f}s, gate={self.match_gate_m:.1f}m): mean={mean_err:.3f} m"
            if errs else
            f"Waiting for obstacle GT + detections... (gate={self.match_gate_m:.1f}m)"
        )
        if goal_err is not None:
            header += f" | goal_err={goal_err:.3f} m"

        self.text.set_text(header + "\n" + "\n".join(lines + goal_lines))

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def main():
    rclpy.init()
    node = ObstacleMapDebugger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
