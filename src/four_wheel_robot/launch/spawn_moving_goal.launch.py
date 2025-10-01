import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Defaults
    default_model = os.path.join(
        get_package_share_directory('four_wheel_robot'),
        'models', 'mobile_goal', 'model.sdf'
    )

    # Args
    model_path   = LaunchConfiguration('model_path')
    entity_name  = LaunchConfiguration('entity_name')
    x = LaunchConfiguration('x');  y = LaunchConfiguration('y');  z = LaunchConfiguration('z')
    R = LaunchConfiguration('R');  P = LaunchConfiguration('P');  Yaw = LaunchConfiguration('Yaw')

    center_x = LaunchConfiguration('center_x'); center_y = LaunchConfiguration('center_y'); radius = LaunchConfiguration('radius')
    v_nom = LaunchConfiguration('v_nom'); w_nom = LaunchConfiguration('w_nom'); jitter_turn = LaunchConfiguration('jitter_turn')

    return LaunchDescription([
        DeclareLaunchArgument('model_path', default_value=default_model),
        DeclareLaunchArgument('entity_name', default_value='mobile_goal'),
        DeclareLaunchArgument('x', default_value='20.0'),
        DeclareLaunchArgument('y', default_value='25.0'),
        DeclareLaunchArgument('z', default_value='0.15'),
        DeclareLaunchArgument('R', default_value='0.0'),
        DeclareLaunchArgument('P', default_value='0.0'),
        DeclareLaunchArgument('Yaw', default_value='0.0'),

        # motion params (zone + behavior)
        DeclareLaunchArgument('center_x', default_value='20.0'),
        DeclareLaunchArgument('center_y', default_value='25.0'),
        DeclareLaunchArgument('radius',   default_value='5.0'),
        DeclareLaunchArgument('v_nom', default_value='0.4'),
        DeclareLaunchArgument('w_nom', default_value='0.8'),
        DeclareLaunchArgument('jitter_turn', default_value='0.25'),

        # Spawn the SDF into Gazebo (Gazebo must be running with libgazebo_ros_factory.so)
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            name='spawn_goal_marker',
            output='screen',
            arguments=[
                '-file', model_path,
                '-entity', entity_name,
                '-x', x, '-y', y, '-z', z,
                '-R', R, '-P', P, '-Y', Yaw
            ],
            parameters=[{'use_sim_time': True}],
        ),

        # Drive the goal with cmd_vel inside the zone
        Node(
            package='scooter_nav',
            executable='moving_goal_twist',
            name='moving_goal_twist',
            output='screen',
            parameters=[{
                'center_x': center_x,
                'center_y': center_y,
                'radius':   radius,
                'v_nom':    v_nom,
                'w_nom':    w_nom,
                'jitter_turn': jitter_turn,
                # if your node has an entity param, set it too:
                'entity_name': entity_name
            }]
        )
    ])
