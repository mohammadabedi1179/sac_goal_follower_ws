from setuptools import find_packages, setup

package_name = 'scooter_nav'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/go_to_zone_and_search.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mohammadabedi',
    maintainer_email='mohammadabedi1179@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'move_entity = scooter_nav.move_entity:main',
            'min_avoid  = scooter_nav.min_avoid:main',
            'go_to_zone_and_search = scooter_nav.go_to_zone_and_search:main',
            'moving_goal = scooter_nav.moving_goal:main',
            'moving_goal_twist = scooter_nav.moving_goal_twist:main',
        ],
    },
)
