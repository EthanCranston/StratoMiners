from setuptools import setup

package_name = 'sensor_fusion'

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
            "hello_world = sensor_fusion.hello_world:main",
            "image_show = sensor_fusion.image_show:main",
            "image_detection = sensor_fusion.image_detection:main"
        ],
    },
)
