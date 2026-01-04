import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, GroupAction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node, PushRosNamespace, SetRemap
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    # robot_state_publisher
    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory("four_wheel_robot"),
                "launch",
                "rsp.launch.py",
            )
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
            "-s", "libgazebo_ros_factory.so",
            world
        ],
        output="screen",
        respawn=True,
    )

    # Spawn robot from robot_description
    spawn_entity = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-topic", "/follower_robot/robot_description",
            "-entity", "my_robot",
            "-x", "0.0", "-y", "0.0", "-z", "0.30", "-Y", "0.7854"
        ],
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    # --- Your existing image_proc groups (unchanged) ---
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
            launch_arguments={'use_sim_time': 'true'}.items()
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
            launch_arguments={'use_sim_time': 'true'}.items()
        ),
    ])

    # --- Controller spawners (Option B) ---
    # NOTE:
    # gazebo_ros2_control creates the controller_manager under your namespace.
    # So we point spawners to /follower_robot/controller_manager explicitly.
    controller_manager_path = "/follower_robot/controller_manager"

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager", controller_manager_path,
        ],
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    wheels_velocity_controller_spawner = Node(
    package="controller_manager",
    executable="spawner",
    arguments=[
        "wheels_velocity_controller",
        "--controller-manager", "/follower_robot/controller_manager",
    ],
    output="screen",
    )


    diff_drive_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "diff_drive_controller",
            "--controller-manager", controller_manager_path,
        ],
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    # Start controllers only AFTER the entity is spawned
    start_controllers_after_spawn = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_entity,
            on_exit=[
                joint_state_broadcaster_spawner,
                wheels_velocity_controller_spawner
            ],
        )
    )
    skid_mapper_node = Node(
    package="four_wheel_robot",
    executable="skid_mapper.py",
    output="screen",
)

    return LaunchDescription([
        rsp,
        gazebo,
        spawn_entity,
        start_controllers_after_spawn,
        skid_mapper_node,
        left_image_proc,
        right_image_proc,
    ])
