from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulation (Gazebo) clock if true",
        ),

        Node(
            package="detectors",
            executable="goal_marker_depth",
            name="goal_marker_depth",
            output="screen",
            emulate_tty=True,
            parameters=[{"use_sim_time": use_sim_time}],
        ),

        Node(
            package="detectors",
            executable="yolo_stereo_detector_disparity",
            name="yolo_stereo_detector_disparity",
            output="screen",
            emulate_tty=True,
            parameters=[{"use_sim_time": use_sim_time}],
        ),

        Node(
            package="detectors",
            executable="stereo_box_depth_from_disparity_IQR_EMA",
            name="stereo_box_depth_from_disparity_IQR_EMA",
            output="screen",
            emulate_tty=True,
            parameters=[{"use_sim_time": use_sim_time}],
        ),
    ])
