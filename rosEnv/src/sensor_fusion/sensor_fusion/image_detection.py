#!/usr/bin/env python3

import rclpy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from rclpy.node import Node
import numpy as np
from ament_index_python.packages import get_package_share_directory

WIDTH = 320
HEIGHT = 320
CONF_THRESHOLD = 0.3
NMS_THRESHOLD = 0.3

#/colcon_workspace/install/darknet_ros/share/darknet_ros/yolo_network_config/weights/yolov2-tiny.weights

class ImageDetection(Node):

    def __init__(self):
        super().__init__("image_detection_node")
        self.get_logger().info("Image Detection Node Started")
        self.subcription = self.create_subscription(
            Image, 
            '/camera/image_raw', 
            self.image_callback, 
            10)
        self.bridge = CvBridge()

        darknet_ros_share_dir = get_package_share_directory('darknet_ros')
        weights = darknet_ros_share_dir + '/yolo_network_config/weights/yolov3.weights'
        cfg = darknet_ros_share_dir + '/yolo_network_config/cfg/yolov3.cfg'
        self.yolo = cv2.dnn.readNetFromDarknet(cfg, weights)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

        self.classes = ['person']
    
    def image_callback(self, image: Image):
        #self.get_logger().info("Image Recieved:" + str(image.header))

        try:
            cv_image = self.bridge.imgmsg_to_cv2(image, image.encoding)
        except CvBridgeError as e:
            self.get_logger().error("CvBridge Error: " + str(e))

        blob = cv2.dnn.blobFromImage(cv_image, 1/255, (WIDTH, HEIGHT), [0,0,0], 1, crop=False)
        self.yolo.setInput(blob)

        layerNames = self.yolo.getLayerNames()
        outputNames = [layerNames[i-1] for i in self.yolo.getUnconnectedOutLayers()]
        # self.get_logger().info(str(outputNames))

        outputs = self.yolo.forward(outputNames)

        # output shapes are (300, 85), (1200, 85), (4800, 85)
        # first 4 values of columns are center x, center y, width, height
        # 5th value is the confidence that there is an object present within the 
        # bounding box. 80 different classes 

        self.findObjects(outputs, cv_image)

        cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
        imS = cv2.resize(cv_image, (1920, 1080))
        cv2.imshow("Window", imS)
        cv2.waitKey(1)
        

    def findObjects(self, outputs, image):
        height, width, channel = image.shape
        bbox = []
        classIds = []
        confidence = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                id = np.argmax(scores)
                conf = scores[id]
                # look for only people with class ID 0
                if conf > CONF_THRESHOLD and id == 0:
                    # get the width and height of the bbox and multiply it by 
                    # the image width and height because it is orginally a percentage
                    w, h = int(detection[2]*width), int(detection[3]*height)
                    # get the center x and y of the bbox and multiply it by the image
                    # width and height. Subtract half the width and height to get the left 
                    # corner of the bbox
                    x, y = int(detection[0]*width - w/2), int(detection[1]*height - h/2)
                    bbox.append([x, y, w, h])
                    classIds.append(id)
                    confidence.append(float(conf))

        # self.get_logger().info(str(len(bbox)))
        # Remove the overlapping boxes, a lower NMS Threshold lowers the amount of boxes
        indices = cv2.dnn.NMSBoxes(bbox, confidence, CONF_THRESHOLD, NMS_THRESHOLD)

        self.draw_bbox(image, bbox, confidence, indices)

    def draw_bbox(self, image, bbox, confidence, indices):
        for i in indices:
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]

            # give corner of bbox and width and height. Provide a color and bbox width
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 5)
            text = "Person " + str(int(confidence[i]*100)) + "%"
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 0, 255), 2)
    


def main(args=None):
    rclpy.init(args=args)

    node = ImageDetection()
    try: 
        rclpy.spin(node)
    except KeyboardInterrupt as e:
        return
    rclpy.shutdown()
