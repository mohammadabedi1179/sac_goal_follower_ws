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
    Node that manages static obstacles in Gazebo.

    Service:
        /reset_random_obstacles  (std_srvs/Trigger)

    Behaviour:
      * On FIRST call:
          - spawn N obstacles.
      * On SUBSEQUENT calls:
          - teleport the SAME obstacles to new poses.

    Placement rule for each obstacle:
      * Use robot pose from /follower_robot/odom
      * Use goal pose  from /goal_marker/odom
      * Place obstacle on a "corridor" between robot and goal:
            - along the segment robot -> goal (with small lateral offset)
            - distance from robot >= min_distance_from_robot  (e.g. 2 m)
            - distance from goal  >= min_distance_from_goal   (e.g. 1 m)
      * This guarantees they are in front of the robot, between robot & goal,
        i.e. inside the useful field of view.
    """

    def __init__(self):
        super().__init__("random_obstacle_spawner")

        # ------- Parameters -------
        self.declare_parameter("num_obstacles", 2)
        self.declare_parameter("min_distance_from_robot", 2.0)
        self.declare_parameter("min_distance_from_goal", 1.0)
        # Max lateral offset (tube half-width) around the robot→goal line
        self.declare_parameter("tube_half_width", 1.5)
        # Safety margin: if robot-goal distance < (min_robot + min_goal + safety_margin),
        # we fall back to a simple placement in front of robot.
        self.declare_parameter("safety_margin", 0.5)

        # For fallback: max radius if we have no goal/robot info
        self.declare_parameter("fallback_radius", 7.0)
        self.declare_parameter("obstacle_prefix", "yolo_obstacle_")

        # Gazebo model names to pick from
        self.declare_parameter(
            "model_names",
            [
                "person_standing",
            ],
        )

        self.num_obstacles = (
            self.get_parameter("num_obstacles").get_parameter_value().integer_value
        )
        self.min_dist_robot = (
            self.get_parameter("min_distance_from_robot")
            .get_parameter_value()
            .double_value
        )
        self.min_dist_goal = (
            self.get_parameter("min_distance_from_goal")
            .get_parameter_value()
            .double_value
        )
        self.tube_half_width = (
            self.get_parameter("tube_half_width").get_parameter_value().double_value
        )
        self.safety_margin = (
            self.get_parameter("safety_margin").get_parameter_value().double_value
        )
        self.fallback_radius = (
            self.get_parameter("fallback_radius").get_parameter_value().double_value
        )
        self.obstacle_prefix = (
            self.get_parameter("obstacle_prefix").get_parameter_value().string_value
        )

        param_val = self.get_parameter("model_names").value
        if isinstance(param_val, list):
            self.model_names: List[str] = [str(x) for x in param_val]
        else:
            self.model_names = [str(param_val)]

        self.get_logger().info(
            f"RandomObstacleSpawner: N={self.num_obstacles}, "
            f"min_dist_robot={self.min_dist_robot}, "
            f"min_dist_goal={self.min_dist_goal}, "
            f"tube_half_width={self.tube_half_width}, "
            f"fallback_radius={self.fallback_radius}"
        )
        self.get_logger().info(f"Models: {self.model_names}")

        # Track which Gazebo entities we spawned
        self.spawned_names: List[str] = []

        # Robot & goal poses
        self.robot_pose: Optional[Tuple[float, float]] = None
        self.goal_pose: Optional[Tuple[float, float]] = None

        # Subscriptions for odometry
        self.create_subscription(ModelStates, "/model_states", self._model_states_cb, 10)
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
        """
        Run a shell command with a configurable timeout.

        Returns True on success (returncode == 0), False otherwise.
        """
        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            if out.returncode != 0:
                self.get_logger().warn(
                    f"Command failed: {' '.join(cmd)}\n"
                    f"stdout:\n{out.stdout}\n"
                    f"stderr:\n{out.stderr}"
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
    ):
        """
        Spawn an obstacle using gazebo_ros spawn_entity.py.

        We retry a few times with a long timeout. Even if spawning ultimately
        fails, we still add the name to spawned_names so that future teleports
        at least TRY to move it (and we see errors if it truly doesn't exist).
        """
        self.get_logger().info(
            f"[OBSTACLES] Spawning {name} (model={model}) at "
            f"x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}"
        )

        cmd = [
            "ros2",
            "run",
            "gazebo_ros",
            "spawn_entity.py",
            "-entity",
            name,
            "-database",
            model,
            "-x",
            str(x),
            "-y",
            str(y),
            "-z",
            "0.0",
            "-Y",
            str(yaw),
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
                f"[OBSTACLES] Failed to spawn {name} after "
                f"{max_attempts} attempts. Will still track it for teleports."
            )

        # IMPORTANT: always track the name so we don't silently lose obstacles
        if name not in self.spawned_names:
            self.spawned_names.append(name)

        return ok


    def _teleport_obstacle(self, name: str, x: float, y: float, yaw: float):
        """
        Move an existing Gazebo entity using /set_entity_state.
        """
        self.get_logger().info(
            f"[OBSTACLES] Teleporting {name} to x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}"
        )
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
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
                + ", z: 0.0}, orientation: {z: "
                + str(qz)
                + ", w: "
                + str(qw)
                + "}}}}"
            ),
        ]
        # Teleports should be quick → shorter timeout is fine
        self._call_cmd(cmd, timeout=10.0)

    # ---------- Sampling logic ----------

    def _sample_between_robot_and_goal(
        self, max_attempts: int = 50
    ) -> Tuple[float, float, float]:
        """
        Sample pose (x, y, yaw) between robot and goal.

        Robot at (rx, ry), goal at (gx, gy).
        Obstacle lies along the segment robot→goal with a small lateral offset.

        Constraints:
            dist(robot, obstacle) >= min_dist_robot
            dist(goal,  obstacle) >= min_dist_goal
        """
        # If no odom yet -> fallback
        if self.robot_pose is None or self.goal_pose is None:
            return self._sample_fallback()

        rx, ry = self.robot_pose
        gx, gy = self.goal_pose

        dx = gx - rx
        dy = gy - ry
        d_rg = math.hypot(dx, dy)

        # If robot & goal are too close, corridor impossible -> fallback
        if d_rg < (self.min_dist_robot + self.min_dist_goal + self.safety_margin):
            self.get_logger().warn(
                f"[OBSTACLES] Robot-goal distance {d_rg:.2f} too small "
                f"for corridor constraints, using fallback."
            )
            return self._sample_fallback()

        ux = dx / d_rg
        uy = dy / d_rg

        # Range of distance along the line from robot
        s_min = self.min_dist_robot
        s_max = d_rg - self.min_dist_goal

        for attempt in range(max_attempts):
            # Choose distance from robot along the line
            s = random.uniform(s_min, s_max)

            # Base point on the line
            px = rx + ux * s
            py = ry + uy * s

            # Lateral offset: limited so it's still in FoV (~tube around line)
            # Perpendicular unit vector
            nx = -uy
            ny = ux

            # Max lateral offset limited by tube_half_width and modest FoV (~30deg)
            max_offset_angle = math.radians(30.0)
            max_offset_geom = s * math.tan(max_offset_angle)
            max_offset = min(self.tube_half_width, max_offset_geom)

            offset = random.uniform(-max_offset, max_offset)
            x = px + nx * offset
            y = py + ny * offset

            # Check distances explicitly (just to be safe)
            d_robot = math.hypot(x - rx, y - ry)
            d_goal = math.hypot(x - gx, y - gy)
            if d_robot < self.min_dist_robot or d_goal < self.min_dist_goal:
                continue

            yaw = random.uniform(-math.pi, math.pi)
            return x, y, yaw

        self.get_logger().warn(
            "[OBSTACLES] Failed to sample valid corridor pose, using fallback."
        )
        return self._sample_fallback()

    def _sample_fallback(self) -> Tuple[float, float, float]:
        """
        Fallback: place anywhere in a circle in front of robot,
        used when we don't yet know robot/goal poses.
        """
        r = random.uniform(self.min_dist_robot, self.fallback_radius)
        theta = random.uniform(-math.pi / 4.0, math.pi / 4.0)  # ±45° in front
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        yaw = random.uniform(-math.pi, math.pi)
        return x, y, yaw

    # ---------- Service handler ----------

    def handle_reset_obstacles(self, request, response):
        self.get_logger().info(
            "[OBSTACLES] Reset request received "
            "(spawn on first call, teleport afterwards)."
        )

        if not self.spawned_names:
            # FIRST CALL → spawn
            self.get_logger().info(
                f"[OBSTACLES] No existing obstacles; spawning {self.num_obstacles}."
            )
            for i in range(self.num_obstacles):
                model = random.choice(self.model_names)
                x, y, yaw = self._sample_between_robot_and_goal()
                entity_name = f"{self.obstacle_prefix}{i}"
                self._spawn_one_obstacle(entity_name, model, x, y, yaw)
        else:
            # SUBSEQUENT CALLS → teleport existing
            self.get_logger().info(
                f"[OBSTACLES] Teleporting {len(self.spawned_names)} existing obstacles."
            )
            for name in self.spawned_names:
                x, y, yaw = self._sample_between_robot_and_goal()
                self._teleport_obstacle(name, x, y, yaw)

        response.success = True
        response.message = (
            f"Obstacles active: {len(self.spawned_names)} "
            "(spawned once, teleported this call)"
        )
        self.get_logger().info("[OBSTACLES] Reset/teleport done")
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
