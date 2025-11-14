cmake_minimum_required(VERSION 3.8)
project(goal_marker_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(xacro REQUIRED)
find_package(robot_state_publisher REQUIRED)

# Install models directory
install(
  DIRECTORY models/
  DESTINATION share/${PROJECT_NAME}/models
  USE_SOURCE_PERMISSIONS
)

# Explicitly install specific files to avoid symlinks
install(
  FILES models/goal_marker/materials/scripts/chassis_pattern.material
  DESTINATION share/${PROJECT_NAME}/models/goal_marker/materials/scripts
)
install(
  FILES models/goal_marker/materials/textures/pattern.png
  DESTINATION share/${PROJECT_NAME}/models/goal_marker/materials/textures
)
install(
  FILES models/goal_marker/model.sdf
  DESTINATION share/${PROJECT_NAME}/models/goal_marker
)

# Install launch files
install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()