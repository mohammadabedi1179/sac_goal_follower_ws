#!/usr/bin/env python3
import math, random, rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

def wrap(a):
    while a > math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

class GoalRandomWalker(Node):
    def __init__(self):
        super().__init__('goal_random_walker')
        self.declare_parameter('cmd_topic', '/goal_marker/cmd_vel')
        self.declare_parameter('odom_topic', '/goal_marker/odom')
        self.declare_parameter('center_x', 5.0)
        self.declare_parameter('center_y', 5.0)
        self.declare_parameter('radius',   5.0)
        self.declare_parameter('v_nom',    0.4)
        self.declare_parameter('w_nom',    0.8)
        self.declare_parameter('dt',       0.05)
        self.declare_parameter('jitter_turn', 0.25)

        self.cx = float(self.get_parameter('center_x').value)
        self.cy = float(self.get_parameter('center_y').value)
        self.R  = float(self.get_parameter('radius').value)
        self.v_nom = float(self.get_parameter('v_nom').value)
        self.w_nom = float(self.get_parameter('w_nom').value)
        self.dt = float(self.get_parameter('dt').value)
        self.jitter = float(self.get_parameter('jitter_turn').value)

        self.cmd_pub = self.create_publisher(Twist, self.get_parameter('cmd_topic').value, 10)
        self.create_subscription(Odometry, self.get_parameter('odom_topic').value, self.odom_cb, 10)

        self.x = self.y = 0.0
        self.yaw = 0.0
        self.have_odom = False
        self.timer = self.create_timer(self.dt, self.step)

    def odom_cb(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        s = 2.0*(q.w*q.z + q.x*q.y); c = 1.0 - 2.0*(q.y*q.y + q.z*q.z)
        self.yaw = math.atan2(s, c)
        self.have_odom = True

    def step(self):
        if not self.have_odom:
            return
        dx, dy = self.x - self.cx, self.y - self.cy
        r = math.hypot(dx, dy)

        cmd = Twist()
        cmd.linear.x = self.v_nom

        if r > self.R:
            desired = math.atan2(self.cy - self.y, self.cx - self.x)
            err = wrap(desired - self.yaw)
            cmd.angular.z = 1.2 * self.w_nom * err
        elif r > 0.8*self.R:
            desired = math.atan2(self.cy - self.y, self.cx - self.x)
            err = wrap(desired - self.yaw)
            cmd.angular.z = 0.8 * self.w_nom * err
            cmd.linear.x  = 0.6 * self.v_nom
        else:
            cmd.angular.z = random.uniform(-self.jitter, self.jitter)

        cmd.angular.z = max(-self.w_nom, min(self.w_nom, cmd.angular.z))
        self.cmd_pub.publish(cmd)

def main():
    rclpy.init()
    node = GoalRandomWalker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
