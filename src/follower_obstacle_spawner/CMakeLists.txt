cmake_minimum_required(VERSION 3.5)
project(follower_obstacle_spawner)

# Use ament_cmake + ament_cmake_python for ament_python-style install
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install the Python package (folder follower_obstacle_spawner/)
ament_python_install_package(${PROJECT_NAME})

# Install the node script into ROS2's lib/ so "ros2 run" can find it
install(
  PROGRAMS
    follower_obstacle_spawner/random_obstacle_spawner.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
