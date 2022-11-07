from geometry_msgs.msg import TransformStamped

import rclpy
from rclpy.node import Node

import os
import yaml
from yaml.loader import SafeLoader

from tf2_ros import TransformBroadcaster
from gb_visual_detection_3d_msgs.msg import BoundingBoxes3d


class TfHuman(Node):

    def __init__(self):
        super().__init__('fixed_frame_tf2_broadcaster')

        # Load YAML file
        cfg = os.path.join(os.path.dirname(os.path.realpath('__file__')), 'src/sensor_fusion/config/sensor_fusion.yaml')
        with open(cfg) as f:
            data = yaml.load(f, Loader=SafeLoader)
            output_bbx3d_topic_ = data['sensor_fusion']['output_bbx3d_topic']
        
        self.tf_broadcaster = TransformBroadcaster(self)
        self.timer = self.create_timer(0.1, self.broadcast_timer_callback)

        self.subscription = self.create_subscription(
            BoundingBoxes3d,
            output_bbx3d_topic_,
            self.bbx_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.boundingboxes = None

    def bbx_callback(self, msg):
        self.boundingboxes = msg

    def broadcast_timer_callback(self):
        if (self.boundingboxes is None):
            return 

        for i, bbx in enumerate(self.boundingboxes.bounding_boxes):
            t = TransformStamped()

            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = self.boundingboxes.header.frame_id
            t.child_frame_id = 'tf_human_' + str(i)

            centerx = (bbx.xmax + bbx.xmin) / 2
            centery = (bbx.ymax + bbx.ymin) / 2
            centerz = (bbx.zmax + bbx.zmin) / 2
            t.transform.translation.x = centerx
            t.transform.translation.y = centery
            t.transform.translation.z = centerz
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0

            self.tf_broadcaster.sendTransform(t)


def main():
    rclpy.init()
    node = TfHuman()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()