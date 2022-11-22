from setuptools import setup

package_name = 'lidar_cv'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lidar_process = lidar_cv.lidar_process:main',
            'find_humans = lidar_cv.find_humans:main'
        ],
    },
)
# setup(
#     name='gb_visual_detection_3d_msgs',
#     version='1.0.0',
#     packages=find_packages(
#         include=('gb_visual_detection_3d_msgs', 'gb_visual_detection_3d_msgs.*')),
# )
