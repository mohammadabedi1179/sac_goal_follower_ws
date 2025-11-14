import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, PushRosNamespace, SetRemap
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
def generate_launch_description():
    # robot_state_publisher
    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory("four_wheel_robot"),
                         "launch", "rsp.launch.py")
        ]),
        launch_arguments={"use_sim_time": "true"}.items(),
    )

    # Gazebo Classic with ONLY the factory system plugin + empty world
    world = os.path.join(
    get_package_share_directory("four_wheel_robot"),
    "worlds",
    "empty_with_state.world",
)

    gazebo = ExecuteProcess(
        cmd=[
            "gazebo", "--verbose",
            "-s", "libgazebo_ros_factory.so",   # keep factory
            world                               # load world that contains the state WORLD plugin
        ],
        output="screen",
        respawn=True,
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
    
    cam_info_from_yaml = Node(
        package="detectors",
        executable="camerainfo_from_yaml",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )
    
    right_image_proc = GroupAction([
        PushRosNamespace('follower_robot/depth_cam/right'),

        SetRemap(src='image',  dst='image_raw'),
        SetRemap(src='camera_info', dst='camera_info'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('image_proc'),
                    'launch',
                    'image_proc.launch.py'
                ])
            ]),
            launch_arguments={
                'use_sim_time': 'true',
            }.items()
        ),
    ])

    left_image_proc = GroupAction([
        PushRosNamespace('follower_robot/depth_cam/left'),

        SetRemap(src='image',  dst='image_raw'),
        SetRemap(src='camera_info', dst='camera_info'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('image_proc'),
                    'launch',
                    'image_proc.launch.py'
                ])
            ]),
            launch_arguments={
                'use_sim_time': 'true',
            }.items()
        ),
    ])

    return LaunchDescription([rsp, 
                              gazebo, 
                              spawn_entity, 
                              #cam_info_from_yaml, 
                              left_image_proc, 
                              right_image_proc])