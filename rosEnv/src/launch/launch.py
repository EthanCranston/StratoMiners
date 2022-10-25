from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    # Launch darknet
    darknet = ExecuteProcess(
        cmd=['ros2', 'launch', 'darknet_ros', 'stratominers.launch.py']
    )

    # Launch darknet 3d
    darknet_3d = ExecuteProcess (
        cmd=['ros2', 'launch', 'darknet_ros_3d', 'darknet_ros_3d.launch.py']
    )

    # Play and loop the bag
    bag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', '-l', '/rosEnv/bags/rosbag2_2022_09_06-14_59_23']
        # cmd=['ros2', 'bag', 'play', '-l', '/rosEnv/bags/rosbag2_2022_09_06-15_07_58']
    )

    # Visualize everything in RViz2
    image_annotaion = ExecuteProcess(
        cmd=['rviz2', '-d', '/rosEnv/src/launch/launch.rviz']
    )

    ld = LaunchDescription()
    ld.add_action(darknet)
    ld.add_action(darknet_3d)
    ld.add_action(bag_play)
    ld.add_action(image_annotaion)

    return ld
