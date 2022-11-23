from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    # Launch lidar_detection
    find_humans = ExecuteProcess(
        cmd=['ros2', 'run', 'lidar_detection', 'find_humans']
    )
    ld = LaunchDescription()
    ld.add_action(find_humans)

    return ld
