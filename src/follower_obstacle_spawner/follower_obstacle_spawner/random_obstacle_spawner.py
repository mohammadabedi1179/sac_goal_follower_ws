#!/usr/bin/env python3
import math
import random
import subprocess
from typing import List

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


class RandomObstacleSpawner(Node):
    """
    Node that (re)spawns random static obstacles in Gazebo.

    - Provides service: /reset_random_obstacles (std_srvs/Trigger)
    - On each call:
        * deletes previously spawned obstacles (if any)
        * spawns N new obstacles within radius R around (0, 0)
    """

    def __init__(self):
        super().__init__("random_obstacle_spawner")

        # Parameters
        self.declare_parameter("num_obstacles", 5)
        self.declare_parameter("radius", 7.0)
        self.declare_parameter("min_distance_from_origin", 3.0)
        self.declare_parameter("obstacle_prefix", "yolo_obstacle_")

        # Gazebo database model names to sample from
        # These must exist in your gazebo_models / GAZEBO_MODEL_PATH
        self.declare_parameter(
            "model_names",
            [
                "person_standing",
                "fire_hydrant"
            ],
        )

        self.num_obstacles = self.get_parameter("num_obstacles").get_parameter_value().integer_value
        self.radius = self.get_parameter("radius").get_parameter_value().double_value
        self.min_dist = self.get_parameter("min_distance_from_origin").get_parameter_value().double_value
        self.obstacle_prefix = self.get_parameter("obstacle_prefix").get_parameter_value().string_value
        param_val = self.get_parameter("model_names").value
        # rclpy already gives us a Python list[str] here
        if isinstance(param_val, list):
            self.model_names: List[str] = [str(x) for x in param_val]
        else:
            # Just in case someone passes a single string
            self.model_names: List[str] = [str(param_val)]

        self.get_logger().info(f"Obstacle model names: {self.model_names}")

        self.spawned_names: List[str] = []

        # Service to reset obstacles
        self.srv = self.create_service(
            Trigger, "reset_random_obstacles", self.handle_reset_obstacles
        )

        self.get_logger().info(
            f"RandomObstacleSpawner ready: N={self.num_obstacles}, "
            f"R={self.radius} m, models={self.model_names}"
        )

    # -------- Helpers for Gazebo CLI calls --------

    def _call_cmd(self, cmd: list) -> bool:
        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10.0
            )
            if out.returncode != 0:
                self.get_logger().warn(
                    f"Command failed: {' '.join(cmd)}\nstdout:\n{out.stdout}\nstderr:\n{out.stderr}"
                )
                return False
            return True
        except subprocess.TimeoutExpired:
            self.get_logger().warn(f"Command timed out: {' '.join(cmd)}")
            return False

    def _delete_existing_obstacles(self):
        """
        Delete all obstacles that follow the naming convention
        obstacle_prefix + index, for i in [0, num_obstacles-1].
        This works even if the node has just started and
        self.spawned_names is empty.
        """
        # 1) Try to delete by index-based names (robust across runs)
        for i in range(self.num_obstacles):
            name = f"{self.obstacle_prefix}{i}"
            cmd = [
                "ros2",
                "service",
                "call",
                "/delete_entity",
                "gazebo_msgs/srv/DeleteEntity",
                f"{{name: '{name}'}}",
            ]
            self.get_logger().info(f"[OBSTACLES] Deleting {name}")
            self._call_cmd(cmd)

        # 2) Also delete anything tracked in spawned_names (just in case)
        for name in self.spawned_names:
            if not name.startswith(self.obstacle_prefix):
                self.get_logger().info(f"[OBSTACLES] Deleting extra {name}")
                cmd = [
                    "ros2",
                    "service",
                    "call",
                    "/delete_entity",
                    "gazebo_msgs/srv/DeleteEntity",
                    f"{{name: '{name}'}}",
                ]
                self._call_cmd(cmd)

        self.spawned_names = []


    def _spawn_one_obstacle(self, name: str, model: str, x: float, y: float, yaw: float):
        """
        Use gazebo_ros spawn_entity.py with -database <model>.
        """
        self.get_logger().info(
            f"[OBSTACLES] Spawning {name} (model={model}) at x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}"
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

        ok = self._call_cmd(cmd)
        if ok:
            self.spawned_names.append(name)
        return ok

    # -------- Service callback --------

    def handle_reset_obstacles(self, request, response):
        self.get_logger().info("[OBSTACLES] Reset request received")

        # 1) Delete old obstacles
        self._delete_existing_obstacles()

        # 2) Spawn new obstacles
        for i in range(self.num_obstacles):
            model = random.choice(self.model_names)

        for i in range(self.num_obstacles):
            model = random.choice(self.model_names)

            # Sample random pose in annulus: 3 m <= radius <= 7 m
            # Use sqrt trick so sampling is uniform in area
            r = math.sqrt(
                random.uniform(self.min_dist ** 2, self.radius ** 2)
            )
            theta = random.uniform(0.0, 2.0 * math.pi)

            x = r * math.cos(theta)  # NO +3 offset
            y = r * math.sin(theta)  # NO +3 offset

            yaw = random.uniform(-math.pi, math.pi)
            entity_name = f"{self.obstacle_prefix}{i}"

            self._spawn_one_obstacle(entity_name, model, x, y, yaw)

        response.success = True
        response.message = f"Spawned {len(self.spawned_names)} obstacles"
        self.get_logger().info("[OBSTACLES] Reset done")
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
