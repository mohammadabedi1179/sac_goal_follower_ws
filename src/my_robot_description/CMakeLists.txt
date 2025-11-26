cmake_minimum_required(VERSION 3.5)
project(my_robot_description)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(urdf REQUIRED)
find_package(xacro REQUIRED)

# Install resources (URDF, meshes, launch)
install(DIRECTORY
  launch
  worlds
  models
  DESTINATION share/${PROJECT_NAME}
)

ament_package()