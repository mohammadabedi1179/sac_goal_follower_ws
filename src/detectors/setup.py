from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'detectors'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # config files
        (os.path.join('share', package_name, 'config'),
         glob('config/*.yaml')),
        # NEW: install msg files
        (os.path.join('share', package_name, 'msg'),
         glob('msg/*.msg')),
        (os.path.join('share', 'detectors', 'launch'),
          glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mohammadabedi',
    maintainer_email='mohammadabedi1179@gmail.com',
    description='Detectos for Follower Robot',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camerainfo_from_yaml = detectors.camerainfo_from_yaml:main',
            'goal_marker_depth = detectors.goal_marker_depth_node:main',
            'yolo_stereo_detector = detectors.yolo_stereo_detector:main',
            'stereo_box_depth = detectors.stereo_box_depth:main',
            'yolo_stereo3d_live = detectors.yolo_stereo3d_live:main',
            'yolo_stereo_detector_IQR_EMA = detectors.yolo_stereo_detector_IQR_EMA:main',
            'yolo_stereo_detector_disparity = detectors.yolo_stereo_detector_disparity:main',
            'yolo_stereo_detector_disparity_light = detectors.yolo_stereo_detector_disparity_light:main',
            'stereo_box_depth_IQR_EMA = detectors.stereo_box_depth_IQR_EMA:mian',
            'stereo_box_depth_disparity = detectors.stereo_box_depth_disparity:main',
            'stereo_box_depth_from_disparity = detectors.stereo_box_depth_from_disparity:main',   
            'stereo_box_depth_from_disparity_IQR_EMA = detectors.stereo_box_depth_from_disparity_IQR_EMA:main',
            'stereo_box_depth_from_disparity_IQR_EMA_synced = detectors.stereo_box_depth_from_disparity_IQR_EMA_synced:main',
            'ultrasonic_obstacle_distance = detectors.ultrasonic_obstacle_distance:main',         
        ],
    },
)
