from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    input_topic_arg = DeclareLaunchArgument(
        'input_topic',
        default_value='/depth_cam/left/image_raw',
        description='Camera image topic'
    )

    return LaunchDescription([
        input_topic_arg,
        Node(
            package='test_codes',
            executable='marker_detector',
            name='red_marker_detector',
            output='screen',
            parameters=[{
                'input_topic': LaunchConfiguration('input_topic'),
                'show_window': True,
                'min_area': 600,
                'lower_red_1': [0, 120, 80],
                'upper_red_1': [12, 255, 255],
                'lower_red_2': [170, 120, 80],
                'upper_red_2': [180, 255, 255],
                'morph_kernel': 5
            }]
        )
    ])
