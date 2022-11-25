from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    # Launch image_detection
    image_detection = ExecuteProcess(
        cmd=['ros2', 'launch', 'darknet_ros', 'stratominers.launch.py']
    )

    # Launch sensor_fusion
    sensor_fusion = ExecuteProcess (
        cmd=['ros2', 'launch', 'darknet_ros_3d', 'darknet_ros_3d.launch.py']
    )

    # Launch lidar_cv
    lidar_detection = ExecuteProcess(
        cmd=['ros2', 'run', 'lidar_detection', 'find_humans']
    )

    # Launch tf_human publisher
    tf_human = Node(
        package='tf_human',
        executable='tf_human',
        name='tf_human_publisher'
    )

    # Visualize everything in RViz2
    rviz = ExecuteProcess(
        cmd=['rviz2', '-d', '/rosEnv/src/launch/launch.rviz']
    )

    ld = LaunchDescription()
    ld.add_action(image_detection)
    ld.add_action(sensor_fusion)
    ld.add_action(lidar_detection)
    ld.add_action(tf_human)
    ld.add_action(rviz)

    return ld
