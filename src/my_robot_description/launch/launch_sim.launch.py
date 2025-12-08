import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, PushRosNamespace, SetRemap, ComposableNodeContainer
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from launch_ros.descriptions import ComposableNode
from launch.conditions import IfCondition, UnlessCondition
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # robot_state_publisher
    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory("my_robot_description"),
                         "launch", "rsp.launch.py")
        ]),
        launch_arguments={"use_sim_time": "true"}.items(),
    )

    # Gazebo Classic with ONLY the factory system plugin + empty world
    world = os.path.join(
    get_package_share_directory("my_robot_description"),
    "worlds",
    "empty_with_state.world",
)

    # 1. Declare the 'gui' argument (keep it for easy launching)
    gui_arg = DeclareLaunchArgument(
        'gui',
        default_value='false',  
        description='Set to "true" to launch the Gazebo client (GUI)'
    )
    gui = LaunchConfiguration('gui')
    
    # ... (RSP and World definition)
    
    # 2. **ALWAYS** Launch the Server (headless core)
    # Note: Using 'gzserver' guarantees the simulation starts headless.
    gazebo_server = ExecuteProcess(
        cmd=[
            "gzserver", 
            "--verbose",
            "-s", "libgazebo_ros_factory.so",
            world                               
        ],
        output="screen",
        respawn=True,
    )
    
    # 3. Conditionally Launch the Client (GUI)
    # This process will only execute if gui == 'true'
    gazebo_client = ExecuteProcess(
        cmd=["gzclient"],
        output="screen",
        condition=IfCondition(gui), # <<< ONLY runs if gui is true
    )

    # Spawn robot from robot_description
    spawn_entity = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=["-topic", "robot_description",
                   "-entity", "my_robot",
                   "-x", "0.0", "-y", "0.0", "-z", "0.30", "-Y", "0.7854"],
        output="screen",
        parameters=[{"use_sim_time": True}],
    )
    name_space = '/follower_robot/depth_cam'
    qos_best_effort = {
        "qos_overrides": {
            "/follower_robot/depth_cam/left/image_raw": {
                "subscription": {
                    "reliability": "best_effort",
                    "history": "keep_last",
                    "depth": 10,
                }
            },
            "/follower_robot/depth_cam/left/camera_info": {
                "subscription": {
                    "reliability": "best_effort",
                    "depth": 5,
                }
            },
            "/follower_robot/depth_cam/right/image_raw": {
                "subscription": {
                    "reliability": "best_effort",
                    "history": "keep_last",
                    "depth": 10,
                }
            },
            "/follower_robot/depth_cam/right/camera_info": {
                "subscription": {
                    "reliability": "best_effort",
                    "depth": 5,
                }
            },
        }
    }

    stereo_and_point_cloud_container = ComposableNodeContainer(
        name='stereo_and_point_cloud_container',
        namespace=name_space,
        package='rclcpp_components',
        executable='component_container_mt', # Multi-threaded container for IPC
        composable_node_descriptions=[
            ComposableNode(
                package='image_proc',
                plugin='image_proc::RectifyNode',
                namespace=name_space,
                name='left_rectify_node',
                parameters=[{
                            'use_sim_time': True,
                            # 'approximate_sync': True,  <-- REMOVE THIS (Use Exact Sync)
                            'queue_size': 10,
                }, qos_best_effort
                ],
                remappings=[
                    ('image', f'{name_space}/left/image_raw'),
                    ('camera_info', f'{name_space}/left/camera_info'),
                    ('image_rect', f'{name_space}/left/image_rect'), 
                ]
            ),
            ComposableNode(
                package='image_proc',
                plugin='image_proc::RectifyNode',
                name='right_rectify_node',
                namespace=name_space,
                parameters=[{
                            'use_sim_time': True,
                            # 'approximate_sync': True,  <-- REMOVE THIS
                            'queue_size': 10,
                }, qos_best_effort,
                ],
                remappings=[
                    ('image', f'{name_space}/right/image_raw'),
                    ('camera_info', f'{name_space}/right/camera_info'),
                    ('image_rect', f'{name_space}/right/image_rect'),
                ]
            ),
            ComposableNode(
                package='stereo_image_proc',
                plugin='stereo_image_proc::DisparityNode',
                name='disparity_node',
                namespace=name_space,
                parameters=[{
                            'use_sim_time': True,
                            # 'approximate_sync': True,  <-- REMOVE THIS
                            'queue_size': 10,
                            },
                    qos_best_effort,
                ],
                remappings=[
                    # FIX: Subscribe to the RECTIFIED images from the nodes above
                    ('left/image_rect', f'{name_space}/left/image_rect'),
                    ('left/camera_info', f'{name_space}/left/camera_info'),
                    ('right/image_rect', f'{name_space}/right/image_rect'),
                    ('right/camera_info', f'{name_space}/right/camera_info'),
                    ('disparity', f'{name_space}/disparity')
                ]
            ),
            ComposableNode(
                package='stereo_image_proc',
                plugin='stereo_image_proc::PointCloudNode', 
                name='point_cloud_node',
                namespace=name_space,
                parameters=[
                            {
                            'use_sim_time': True,
                            # 'approximate_sync': True,  <-- REMOVE THIS
                            'queue_size': 10,
                            }
                            ],
                remappings=[
                    ('left/image_rect_color', f'{name_space}/left/image_rect'),
                    ('left/camera_info', f'{name_space}/left/camera_info'),
                    ('right/camera_info', f'{name_space}/right/camera_info'),
                    ('points2', f'{name_space}/points2'),
                    ('disparity', f'{name_space}/disparity')
                ]
            ),
        ],
        output='screen',
    )

    return LaunchDescription([
                              gui_arg,
                              rsp, 
                              gazebo_server,         # Runs every time
                              gazebo_client,         # Runs ONLY if gui:=true
                              spawn_entity,
                              stereo_and_point_cloud_container
                             ])