#!/usr/bin/env python3
import math
import random
import subprocess
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates


class RandomObstacleSpawner(Node):
    """
    Node that manages obstacles in Gazebo.

    Service:
        /reset_random_obstacles  (std_srvs/Trigger)

    Behaviour:
      * On FIRST call (episode 1):
          - Spawn 7 WALKING persons anywhere in the working zone circle.
          - Spawn 3 STANDING persons inside the robot→goal sector:
              · Sector center: robot position
              · Angular range: ±30° around the robot→goal bearing
              · Radial range: min_distance_from_robot (1.5 m) to
                              (robot-goal dist − min_distance_from_goal)
          - Walking persons are NEVER moved again.

      * On SUBSEQUENT calls:
          - Only the 3 STANDING persons are teleported to new sector
            positions (because the goal marker has moved).
          - Walking persons remain where they are.
    """

    # ---- Obstacle counts ----
    NUM_WALKING  = 5   # spawned once, never moved
    NUM_STANDING = 1   # re-teleported every episode

    def __init__(self):
        super().__init__("random_obstacle_spawner")

        # ------- Parameters -------
        self.declare_parameter("min_distance_from_robot", 1.5)
        self.declare_parameter("min_distance_from_goal",  1.0)
        # Sector half-angle around the robot→goal line (degrees → radians)
        self.declare_parameter("sector_half_angle_deg", 30.0)
        # Working zone radius for walking persons
        self.declare_parameter("working_zone_radius", 8.0)
        # Safety margin: if robot-goal dist < (min_robot + min_goal + safety_margin)
        # we fall back to simple placement.
        self.declare_parameter("safety_margin", 0.5)
        # Fallback radius when robot/goal pose is unknown
        self.declare_parameter("fallback_radius", 7.0)
        self.declare_parameter("obstacle_prefix", "yolo_obstacle_")

        # Walking-person model name
        self.declare_parameter("walking_model_name", "person_walking")
        # Standing-person model name
        self.declare_parameter("standing_model_name", "person_standing")

        self.min_dist_robot = (
            self.get_parameter("min_distance_from_robot")
            .get_parameter_value().double_value
        )
        self.min_dist_goal = (
            self.get_parameter("min_distance_from_goal")
            .get_parameter_value().double_value
        )
        self.sector_half_angle = math.radians(
            self.get_parameter("sector_half_angle_deg")
            .get_parameter_value().double_value
        )
        self.working_zone_radius = (
            self.get_parameter("working_zone_radius")
            .get_parameter_value().double_value
        )
        self.safety_margin = (
            self.get_parameter("safety_margin")
            .get_parameter_value().double_value
        )
        self.fallback_radius = (
            self.get_parameter("fallback_radius")
            .get_parameter_value().double_value
        )
        self.obstacle_prefix  = (
            self.get_parameter("obstacle_prefix")
            .get_parameter_value().string_value
        )
        self.walking_model  = (
            self.get_parameter("walking_model_name")
            .get_parameter_value().string_value
        )
        self.standing_model = (
            self.get_parameter("standing_model_name")
            .get_parameter_value().string_value
        )

        self.get_logger().info(
            f"RandomObstacleSpawner: "
            f"walking={self.NUM_WALKING}×{self.walking_model}, "
            f"standing={self.NUM_STANDING}×{self.standing_model}, "
            f"min_dist_robot={self.min_dist_robot}, "
            f"min_dist_goal={self.min_dist_goal}, "
            f"sector_half_angle={math.degrees(self.sector_half_angle):.1f}°, "
            f"working_zone_radius={self.working_zone_radius}"
        )

        # ---- Entity name lists ----
        # Walking persons: indices 0 … NUM_WALKING-1
        self.walking_names: List[str] = [
            f"{self.obstacle_prefix}{i}" for i in range(self.NUM_WALKING)
        ]
        # Standing persons: indices NUM_WALKING … NUM_WALKING+NUM_STANDING-1
        self.standing_names: List[str] = [
            f"{self.obstacle_prefix}{self.NUM_WALKING + i}"
            for i in range(self.NUM_STANDING)
        ]

        # Track which ones have actually been spawned in Gazebo
        self.walking_spawned:  bool = False
        self.standing_spawned: bool = False

        # Robot & goal poses
        self.robot_pose: Optional[Tuple[float, float]] = None
        self.goal_pose:  Optional[Tuple[float, float]] = None

        # Subscriptions
        self.create_subscription(
            ModelStates, "/model_states", self._model_states_cb, 10
        )
        self.create_subscription(
            Odometry, "/goal_marker/odom", self._goal_odom_cb, 10
        )

        # Service
        self.srv = self.create_service(
            Trigger, "reset_random_obstacles", self.handle_reset_obstacles
        )

        self.get_logger().info("RandomObstacleSpawner ready.")

    # ---------- Callbacks ----------

    def _model_states_cb(self, msg: ModelStates):
        try:
            i = msg.name.index("my_robot")
        except ValueError:
            return
        p = msg.pose[i].position
        self.robot_pose = (p.x, p.y)

    def _goal_odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self.goal_pose = (p.x, p.y)

    # ---------- Gazebo helpers ----------

    def _call_cmd(self, cmd: list, timeout: float = 30.0) -> bool:
        """Run a shell command. Returns True on success (returncode == 0)."""
        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            if out.returncode != 0:
                self.get_logger().warn(
                    f"Command failed: {' '.join(cmd)}\n"
                    f"stdout:\n{out.stdout}\nstderr:\n{out.stderr}"
                )
                return False
            return True
        except subprocess.TimeoutExpired:
            self.get_logger().warn(
                f"Command timed out after {timeout:.1f}s: {' '.join(cmd)}"
            )
            return False

    def _spawn_one_obstacle(
        self,
        name: str,
        model: str,
        x: float,
        y: float,
        yaw: float,
        max_attempts: int = 3,
    ) -> bool:
        """
        Spawn an obstacle using gazebo_ros spawn_entity.py.
        Retries up to max_attempts times.
        """
        self.get_logger().info(
            f"[OBSTACLES] Spawning {name} (model={model}) at "
            f"x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}"
        )

        cmd = [
            "ros2", "run", "gazebo_ros", "spawn_entity.py",
            "-entity",   name,
            "-database", model,
            "-x", str(x),
            "-y", str(y),
            "-z", "0.0",
            "-Y", str(yaw),
        ]

        ok = False
        for attempt in range(max_attempts):
            self.get_logger().info(
                f"[OBSTACLES] spawn {name} attempt {attempt + 1}/{max_attempts}"
            )
            ok = self._call_cmd(cmd, timeout=30.0)
            if ok:
                break

        if not ok:
            self.get_logger().error(
                f"[OBSTACLES] Failed to spawn {name} after {max_attempts} attempts."
            )
        return ok

    def _teleport_obstacle(self, name: str, x: float, y: float, yaw: float):
        """Move an existing Gazebo entity using /set_entity_state."""
        self.get_logger().info(
            f"[OBSTACLES] Teleporting {name} to x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}"
        )
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        cmd = [
            "ros2", "service", "call",
            "/set_entity_state",
            "gazebo_msgs/srv/SetEntityState",
            (
                "{state: {name: '" + name +
                "', pose: {position: {x: " + str(x) +
                ", y: " + str(y) +
                ", z: 0.0}, orientation: {z: " + str(qz) +
                ", w: " + str(qw) + "}}}}"
            ),
        ]
        self._call_cmd(cmd, timeout=10.0)

    # ---------- Pose samplers ----------

    def _sample_working_zone(self, max_attempts: int = 100) -> Tuple[float, float, float]:
        """
        Sample (x, y, yaw) uniformly inside the working-zone circle centred on
        the robot, keeping safety distances from robot and goal.
        Falls back to _sample_fallback() if poses are unknown or sampling fails.
        """
        if self.robot_pose is None:
            return self._sample_fallback()

        rx, ry = self.robot_pose
        gx, gy = self.goal_pose if self.goal_pose is not None else (None, None)

        for _ in range(max_attempts):
            # Uniform sampling inside circle of radius working_zone_radius
            r     = self.working_zone_radius * math.sqrt(random.random())
            theta = random.uniform(0.0, 2.0 * math.pi)
            x = rx + r * math.cos(theta)
            y = ry + r * math.sin(theta)

            if math.hypot(x - rx, y - ry) < self.min_dist_robot:
                continue
            if gx is not None and math.hypot(x - gx, y - gy) < self.min_dist_goal:
                continue

            yaw = random.uniform(-math.pi, math.pi)
            return x, y, yaw

        self.get_logger().warn(
            "[OBSTACLES] Working-zone sampling failed, using fallback."
        )
        return self._sample_fallback()

    def _sample_sector(self, max_attempts: int = 200) -> Tuple[float, float, float]:
        """
        Sample (x, y, yaw) inside the robot→goal sector:
            - Angular range : ±sector_half_angle around the robot→goal bearing
            - Radial range  : [min_dist_robot, robot_goal_dist - min_dist_goal]
            - Also enforces min distance from goal

        Falls back to _sample_working_zone() if poses are unknown or the
        robot-goal corridor is too short.
        """
        if self.robot_pose is None or self.goal_pose is None:
            self.get_logger().warn(
                "[OBSTACLES] Sector sample: no robot/goal pose – using working zone."
            )
            return self._sample_working_zone()

        rx, ry = self.robot_pose
        gx, gy = self.goal_pose

        robot_goal_dist = math.hypot(gx - rx, gy - ry)
        bearing_to_goal = math.atan2(gy - ry, gx - rx)

        r_min = self.min_dist_robot
        r_max = robot_goal_dist - self.min_dist_goal

        if r_max - r_min < self.safety_margin:
            self.get_logger().warn(
                f"[OBSTACLES] Robot-goal dist ({robot_goal_dist:.2f} m) too small "
                f"for sector placement – falling back to working zone."
            )
            return self._sample_working_zone()

        for _ in range(max_attempts):
            # Uniform radial sample
            r = random.uniform(r_min, r_max)
            # Uniform angular sample within sector
            alpha = random.uniform(
                bearing_to_goal - self.sector_half_angle,
                bearing_to_goal + self.sector_half_angle,
            )
            x = rx + r * math.cos(alpha)
            y = ry + r * math.sin(alpha)

            # Double-check distances (the bounds already guarantee them for
            # the nominal case, but explicit checks guard edge cases)
            if math.hypot(x - rx, y - ry) < self.min_dist_robot:
                continue
            if math.hypot(x - gx, y - gy) < self.min_dist_goal:
                continue

            yaw = random.uniform(-math.pi, math.pi)
            return x, y, yaw

        self.get_logger().warn(
            "[OBSTACLES] Sector sampling failed after max attempts, using working zone."
        )
        return self._sample_working_zone()

    def _sample_fallback(self) -> Tuple[float, float, float]:
        """Fallback: place anywhere in a cone in front of the world origin."""
        r     = random.uniform(self.min_dist_robot, self.fallback_radius)
        theta = random.uniform(-math.pi / 4.0, math.pi / 4.0)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        yaw = random.uniform(-math.pi, math.pi)
        return x, y, yaw

    # ---------- Service handler ----------

    def handle_reset_obstacles(self, request, response):
        self.get_logger().info(
            "[OBSTACLES] Reset request received."
        )

        if not self.walking_spawned and not self.standing_spawned:
            # ── FIRST EPISODE ──────────────────────────────────────────────
            self.get_logger().info(
                f"[OBSTACLES] First call: spawning "
                f"{self.NUM_WALKING} walking + {self.NUM_STANDING} standing persons."
            )

            # --- 7 walking persons in the working zone ---
            for name in self.walking_names:
                x, y, yaw = self._sample_working_zone()
                self._spawn_one_obstacle(name, self.walking_model, x, y, yaw)
            self.walking_spawned = True

            # --- 3 standing persons in the robot→goal sector ---
            for name in self.standing_names:
                x, y, yaw = self._sample_sector()
                self._spawn_one_obstacle(name, self.standing_model, x, y, yaw)
            self.standing_spawned = True

        else:
            # ── SUBSEQUENT EPISODES ────────────────────────────────────────
            # Walking persons are deliberately left in place.
            # Only standing persons are teleported (goal has moved).
            self.get_logger().info(
                f"[OBSTACLES] Subsequent call: teleporting "
                f"{self.NUM_STANDING} standing persons to new sector positions."
            )
            for name in self.standing_names:
                x, y, yaw = self._sample_sector()
                self._teleport_obstacle(name, x, y, yaw)

        response.success = True
        response.message = (
            f"Walking persons: {self.NUM_WALKING} (static). "
            f"Standing persons: {self.NUM_STANDING} (repositioned this episode)."
        )
        self.get_logger().info("[OBSTACLES] Reset done.")
        return response


def main(args=None):
    rclpy.init(args=args)
    node = RandomObstacleSpawner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down RandomObstacleSpawner")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
