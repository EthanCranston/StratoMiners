#!/usr/bin/env python3

import rclpy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from rclpy.node import Node

class ImageShow(Node):

    def __init__(self):
        super().__init__("image_show_node")
        self.get_logger().info("Image Show Node Started")
        self.subcription = self.create_subscription(
            Image, 
            '/darknet_ros/detection_image', 
            self.image_callback, 
            10)
        self.bridge = CvBridge()
    
    def image_callback(self, image: Image):
        #self.get_logger().info("Image Recieved:" + str(image.header))
        self.get_logger().info(f"Width: {image.width}")
        self.get_logger().info(f"Height: {image.height}")

        try:
            cv_image = self.bridge.imgmsg_to_cv2(image, image.encoding)
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")

        cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
        imS = cv2.resize(cv_image, (960, 540))
        cv2.imshow("Window", imS)
        cv2.waitKey(3)
        



def main(args=None):
    rclpy.init(args=args)

    node = ImageShow()
    try: 
        rclpy.spin(node)
    except KeyboardInterrupt as e:
        return
    rclpy.shutdown()
