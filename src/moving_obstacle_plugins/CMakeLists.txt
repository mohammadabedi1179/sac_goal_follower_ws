cmake_minimum_required(VERSION 3.5)
project(moving_obstacle_plugins)

find_package(ament_cmake REQUIRED)

# CRITICAL: Force CMake to use SYSTEM Boost, not /usr/local
set(CMAKE_IGNORE_PATH "/usr/local/include;/usr/local/lib")
set(Boost_NO_SYSTEM_PATHS OFF)
set(BOOST_ROOT "/usr")
set(BOOST_INCLUDEDIR "/usr/include")
set(BOOST_LIBRARYDIR "/usr/lib/x86_64-linux-gnu")

find_package(gazebo REQUIRED)
find_package(TBB REQUIRED)

add_library(moving_obstacle_plugin SHARED
  src/moving_obstacle_plugin.cpp
)

# Force system boost headers
target_include_directories(moving_obstacle_plugin PRIVATE
  include
  ${GAZEBO_INCLUDE_DIRS}
)

target_compile_options(moving_obstacle_plugin PRIVATE
  ${GAZEBO_CXX_FLAGS}
)

# DON'T link boost_system - it's a header-only stub in Boost 1.74
# The symbols come from boost_filesystem which Gazebo already links
target_link_libraries(moving_obstacle_plugin
  ${GAZEBO_LIBRARIES}
  TBB::tbb
)

# Set RPATH to find libraries at runtime
set_target_properties(moving_obstacle_plugin PROPERTIES
  INSTALL_RPATH "/lib/x86_64-linux-gnu;/usr/lib/x86_64-linux-gnu"
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

install(TARGETS moving_obstacle_plugin
  LIBRARY DESTINATION lib
)

ament_package()
