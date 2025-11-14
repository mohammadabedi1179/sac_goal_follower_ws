from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('left_image_topic',  default_value='/depth_cam/left/image_raw'),
        DeclareLaunchArgument('right_image_topic', default_value='/depth_cam/right/image_raw'),
        DeclareLaunchArgument('left_info_topic',   default_value='/depth_cam/left/camera_info'),
        DeclareLaunchArgument('right_info_topic',  default_value='/depth_cam/right/camera_info'),

        Node(
            package='test_codes',
            executable='stereo_marker_depth',
            name='stereo_marker_depth',
            output='screen',
            parameters=[{
                'left_image_topic':  LaunchConfiguration('left_image_topic'),
                'right_image_topic': LaunchConfiguration('right_image_topic'),
                'left_info_topic':   LaunchConfiguration('left_info_topic'),
                'right_info_topic':  LaunchConfiguration('right_info_topic'),
                'show_window': True,
                'min_area': 600,
                'morph_kernel': 5,
                'lower_red_1': [0, 120, 80],
                'upper_red_1': [12, 255, 255],
                'lower_red_2': [170, 120, 80],
                'upper_red_2': [180, 255, 255],
                'num_disparities': 128,
                'block_size': 5,
                # Set this ONLY if your CameraInfo lacks P_right[0,3]
                # 'baseline_m': 0.12,
            }]
        )
    ])
