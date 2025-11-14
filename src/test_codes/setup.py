from setuptools import setup

package_name = 'test_codes'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/marker_detector.launch.py', 'launch/stereo_marker_depth.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='Sandbox CV nodes (red marker detection)',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'marker_detector = test_codes.marker_detector:main',
            'stereo_marker_depth = test_codes.stereo_marker_depth:main',
        ],
    },
)
