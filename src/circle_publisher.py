#!/usr/bin/env python3
"""
circle_publisher.py
Publishes a circular trajectory as PoseStamped to /command/pose
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from math import sin, cos

class CirclePublisherNode(Node):
    def __init__(self):
        super().__init__('circle_publisher')

        # Publisher to the same topic as in C++ code
        self.publisher_ = self.create_publisher(PoseStamped, 'command/pose', 10)

        # Timer at 100 Hz (0.01 s period)
        self.timer = self.create_timer(0.01, self.publish_circle_pose)

        # State variable
        self.angle = 0.0
        self.radius = 2.0

        self.get_logger().info("CirclePublisherNode started, publishing circular trajectory...")

    def publish_circle_pose(self):
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.header.frame_id = "base_link"   # Change if needed

        pose_stamped.pose.position.x = self.radius * cos(self.angle)
        pose_stamped.pose.position.y = self.radius * sin(self.angle)
        pose_stamped.pose.position.z = 2.0

        # Simple orientation: identity quaternion
        pose_stamped.pose.orientation.w = 1.0
        pose_stamped.pose.orientation.x = 0.0
        pose_stamped.pose.orientation.y = 0.0
        pose_stamped.pose.orientation.z = 0.0

        self.publisher_.publish(pose_stamped)

        self.angle += 0.01   # controls angular speed


def main(args=None):
    rclpy.init(args=args)
    node = CirclePublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()