from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='scooter_nav',
            executable='go_to_zone_and_search',
            name='go_to_zone_and_search',
            output='screen',
            parameters=[
                {'goal_x': 20.0},
                {'goal_y': 25.0},
                {'zone_radius': 10.0},
                {'v_max': 1.0},
                {'w_max': 0.7},
            ],
        )
    ])
