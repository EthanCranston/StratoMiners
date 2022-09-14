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
            '/camera/image_raw', 
            self.image_callback, 
            10)
        self.bridge = CvBridge()
    
    def image_callback(self, image: Image):
        #self.get_logger().info("Image Recieved:" + str(image.header))

        try:
            cv_image = self.bridge.imgmsg_to_cv2(image, image.encoding)
        except CvBridgeError as e:
            self.get_logger().error("CvBridge Error: " + str(e))

        cv2.imshow("Window", cv_image)
        cv2.waitKey(3)
        



def main(args=None):
    rclpy.init(args=args)

    node = ImageShow()
    try: 
        rclpy.spin(node)
    except KeyboardInterrupt as e:
        return
    rclpy.shutdown()
