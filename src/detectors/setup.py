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
        ],
    },
)
