import sys
import time

import rclpy
from a3_description.msg import LandmarkObservations, WheelIncrement
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.time import Time
from sensor_msgs.msg import Imu

DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
TOPICS = [('/imu', Imu), ('/wheel_increments', WheelIncrement),
          ('/landmarks', LandmarkObservations), ('/ekf/odom', Odometry),
          ('/odom_own', Odometry)]


class Measure(Node):
    def __init__(self):
        super().__init__('measure_p5c')
        self.stat = {n: dict(count=0, delays=[], last=None) for n, _ in TOPICS}
        q = QoSProfile(depth=1000)
        for name, typ in TOPICS:
            self.create_subscription(typ, name, lambda m, n=name: self.cb(n, m), q)

    def cb(self, name, m):
        now = self.get_clock().now()
        d = (now - Time.from_msg(m.header.stamp)).nanoseconds * 1e-9
        s = self.stat[name]
        s['count'] += 1
        s['delays'].append(d)
        s['last'] = d


def main():
    rclpy.init()
    node = Measure()
    while node.get_clock().now().nanoseconds == 0:
        rclpy.spin_once(node, timeout_sec=0.1)
    sim0 = node.get_clock().now().nanoseconds * 1e-9
    wall0 = time.time()
    for s in node.stat.values():
        s['count'], s['delays'] = 0, []
    while time.time() - wall0 < DURATION:
        rclpy.spin_once(node, timeout_sec=0.05)
    wall = time.time() - wall0
    sim = node.get_clock().now().nanoseconds * 1e-9 - sim0
    print('wall %.2f s  sim %.2f s  real-time factor %.3f' % (wall, sim, sim / wall))
    print('%-18s %10s %10s %12s %12s %12s' % ('topic', 'Hz(wall)', 'Hz(sim)', 'last delay', 'mean delay', 'max delay'))
    for name, _ in TOPICS:
        s = node.stat[name]
        dl = s['delays']
        print('%-18s %10.1f %10.1f %10.0f ms %10.0f ms %10.0f ms' % (
            name, s['count'] / wall, s['count'] / sim, 1e3 * s['last'],
            1e3 * sum(dl) / len(dl), 1e3 * max(dl)))
    node.destroy_node()
    rclpy.shutdown()


main()
