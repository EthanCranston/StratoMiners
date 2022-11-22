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

    # Launch lidar_cv
    lidar_cv = ExecuteProcess(
        cmd=['ros2', 'run', 'lidar_cv', 'find_humans']
    )

    # Launch tf_human publisher
    tf_human = Node(
        package='sensor_fusion',
        executable='tf_human',
        name='tf_human_publisher'
    )

    # Visualize everything in RViz2
    image_annotaion = ExecuteProcess(
        cmd=['rviz2', '-d', '/rosEnv/src/launch/launch.rviz']
    )

    ld = LaunchDescription()
    ld.add_action(darknet)
    ld.add_action(darknet_3d)
    ld.add_action(lidar_cv)
    ld.add_action(tf_human)
    ld.add_action(image_annotaion)

    return ld
