# This script serves as a launch file for darknet
# It is designed to work with the raw image streamimport os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Generates a launch description
# Uses a modified yolov7-tiny dataset that only outputs persons
def generate_launch_description():
    darknet_ros_share_dir = get_package_share_directory('darknet_ros')

    # Use raw camera image and modified yolov7-tiny weights
    # The weights contain dont-show names for all non-persons
    # This is a workaround for having to retrain our entire dataset
    image = LaunchConfiguration('image', default='image_raw')
    yolo_weights_path = LaunchConfiguration(
        'yolo_weights_path', default=darknet_ros_share_dir + '/yolo_network_config/weights')
    yolo_config_path = LaunchConfiguration(
        'yolo_config_path', default=darknet_ros_share_dir + '/yolo_network_config/cfg')
    ros_param_file = LaunchConfiguration(
        'ros_param_file', default=darknet_ros_share_dir + 'config/ros-stratominers.yaml')
    network_param_file = LaunchConfiguration(
        'network_param_file', default=darknet_ros_share_dir + 'config/yolov7-tiny-stratominers.yaml')

    # Declare images and weights
    declare_image_cmd = DeclareLaunchArgument(
        'image',
        default_value='image_raw',
        description='Image topic')
    declare_yolo_weights_path_cmd = DeclareLaunchArgument(
        'yolo_weights_path',
        default_value=darknet_ros_share_dir + '/yolo_network_config/weights',
        description='Path to yolo weights')
    declare_yolo_config_path_cmd = DeclareLaunchArgument(
        'yolo_config_path',
        default_value=darknet_ros_share_dir + '/yolo_network_config/cfg',
        description='Path to yolo config')
    declare_ros_param_file_cmd = DeclareLaunchArgument(
        'ros_param_file',
        default_value=darknet_ros_share_dir + '/config/ros-stratominers.yaml',
        description='Path to file with ROS related config')
    declare_network_param_file_cmd = DeclareLaunchArgument(
        'network_param_file',
        default_value=darknet_ros_share_dir + '/config/yolov7-tiny-stratominers.yaml',
        description='Path to file with network param file')

    # Make node out of raw image stream
    darknet_ros_cmd = Node(
        package='darknet_ros',
        executable='darknet_ros',
        name='darknet_ros',
        output='screen',
        parameters=[ros_param_file, network_param_file,
                    {
                        "config_path": yolo_config_path,
                        "weights_path": yolo_weights_path,
                    },
                    ])

    # Instantiate launch description
    ld = LaunchDescription()

    # Add actions to instantiation
    ld.add_action(declare_image_cmd)
    ld.add_action(declare_yolo_weights_path_cmd)
    ld.add_action(declare_yolo_config_path_cmd)
    ld.add_action(declare_ros_param_file_cmd)
    ld.add_action(declare_network_param_file_cmd)
    ld.add_action(darknet_ros_cmd)

    # Return the instantiation for launch
    return ld