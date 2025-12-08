from setuptools import setup

package_name = "follower_obstacle_spawner"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="mohammadabedi",
    maintainer_email="you@example.com",
    description="Random obstacle spawner for follower robot environment",
    license="TODO",  # set your license
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "random_obstacle_spawner = follower_obstacle_spawner.random_obstacle_spawner:main",
        ],
    },
)
