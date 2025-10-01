from setuptools import find_packages, setup
import site, sys

package_name = 'sac_goal_follower'
site.addsitedir("/home/mohammadabedi/anaconda3/envs/sac_env/lib/python3.10/site-packages")

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/train_only.launch.py'])
    ],
    install_requires=[
        'setuptools',
        'gymnasium',
        'stable-baselines3',
        'numpy',
        'opencv-python'
        ],
    zip_safe=True,
    maintainer='mohammadabedi',
    maintainer_email='mohammadabedi1179@gmail.com',
    description='SAC agent to follow moving goal_marker using camera distance & bearing',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'train_sac = sac_goal_follower.train_sac:main'
        ],
    },
)