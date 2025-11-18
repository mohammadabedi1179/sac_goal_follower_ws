from setuptools import find_packages, setup

package_name = 'sac_goal_follower'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/train_only.launch.py']),
        # Install the Python module files
        ('lib/' + package_name, ['sac_goal_follower/__init__.py', 'sac_goal_follower/goal_env.py', 'sac_goal_follower/train_sac.py'])
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
    entry_points={
        'console_scripts': [
            'train_sac = sac_goal_follower.train_sac:main',
            'test_disparity = sac_goal_follower.test_disparity:main',
            'test_sac = sac_goal_follower.test_sac:main',
        ],
    },
)