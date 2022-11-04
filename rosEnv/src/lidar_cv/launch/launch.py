from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    # Launch darknet
    find_humans = ExecuteProcess(
        cmd=['ros2', 'run', 'lidar_cv', 'find_humans']
    )

    # Play and loop the bag
    bag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', '-l', '/rosEnv/bags/rosbag2_2022_09_06-15_07_58']
    )

    # Visualize everything in RViz2
    # image_annotaion = ExecuteProcess(
    #     cmd=['rviz2', '-d', '/rosEnv/src/launch/launch.rviz']
    # )

    ld = LaunchDescription()
    ld.add_action(find_humans)
    ld.add_action(bag_play)

    return ld