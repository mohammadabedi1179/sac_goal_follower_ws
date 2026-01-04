cmake_minimum_required(VERSION 3.5)
project(my_robot_description)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(urdf REQUIRED)
find_package(xacro REQUIRED)

# Install resources (URDF, meshes, launch)
install(DIRECTORY
  launch
  worlds
  models
  config
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  my_robot_description/skid_mapper.py
  DESTINATION lib/${PROJECT_NAME}
)
ament_package()