from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    # launch darknet
    darknet = ExecuteProcess(
        cmd=['ros2', 'launch', 'darknet_ros', 'stratominers.launch.py']
    )

    # Play the bag
    bag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', '/rosEnv/bags/rosbag2_2022_09_06-14_59_23']
    )

    image_annotaion = Node(
        package='sensor_fusion',
        executable='image_show',
        name='image_show',
        output='screen'
    )

    ld = LaunchDescription()
    ld.add_action(darknet)
    ld.add_action(bag_play)
    ld.add_action(image_annotaion)

    return ld
