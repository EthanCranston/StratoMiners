#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

class HelloWorld(Node):

    def __init__(self):
        super().__init__("hello_world_node")
        self.get_logger().info("Hello World!")

def main(args=None):
    rclpy.init(args=args)

    node = HelloWorld()
    rclpy.shutdown()