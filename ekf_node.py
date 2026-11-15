#!/usr/bin/env python3

import math
import os

import numpy as np
import rclpy
import yaml
from a3_description.msg import LandmarkObservations, WheelIncrement
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.time import Time
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64
from tf2_ros import Buffer, TransformBroadcaster, TransformListener

from ekf import Ekf

NS = 10 ** 9


def stamp_ns(s):
    return s.sec * NS + s.nanosec


def ns_to_msg(ns):
    return Time(nanoseconds=int(ns)).to_msg()


def fill_odom(msg, frame, child, ns, x, P):
    """Pose (x, y, theta) and its 3 x 3 position-heading covariance into a nav_msgs/Odometry."""
    msg.header.stamp = ns_to_msg(ns)
    msg.header.frame_id, msg.child_frame_id = frame, child
    msg.pose.pose.position.x, msg.pose.pose.position.y = float(x[0]), float(x[1])
    msg.pose.pose.orientation.z = math.sin(0.5 * x[2])
    msg.pose.pose.orientation.w = math.cos(0.5 * x[2])
    cov = [0.0] * 36
    for i, a in enumerate((0, 1, 5)):          # x, y, yaw
        for j, b in enumerate((0, 1, 5)):
            cov[6 * a + b] = float(P[i, j])
    msg.pose.covariance = cov


class EkfNode(Node):
    def __init__(self):
        super().__init__('ekf')
        # ---- parameters: change the defaults, or pass  -p name:=value ----------------------
        self.declare_parameter('init_x', -3.5)        # initial pose in the map frame (Problem 4)
        self.declare_parameter('init_y', -3.0)
        self.declare_parameter('init_yaw', 0.0)
        self.declare_parameter('b0_hat', -0.01222)
        self.declare_parameter('var_b0', 0.0042 ** 2)
        self.declare_parameter('sigma_n', 0.015)
        self.declare_parameter('sigma_g', 0.0042)
        self.declare_parameter('tau', 3.4)
        self.declare_parameter('k1', 1e-3)            # wheel noise, Var(ds) = k1 |ds|
        self.declare_parameter('k2', 1e-3)            # Var(dtheta) = k2 |dtheta|
        self.declare_parameter('sigma_r', 0.02)       # detection noise
        self.declare_parameter('sigma_beta', 0.01)
        self.declare_parameter('gate_chi2', 9.21)     # 99 % of chi-square with 2 degrees of freedom
        self.declare_parameter('stop_eps', 1e-6)      # a tick is "still" if |ds|, |dtheta| are below
        self.declare_parameter('stop_time', 0.5)      # s of still ticks before a stop is declared
        g = lambda n: self.get_parameter(n).value
        params = {n: g(n) for n in ('sigma_n', 'sigma_g', 'tau', 'k1', 'k2', 'sigma_r')}
        params.update(sigma_b=g('sigma_beta'), dt=0.01, cam_x=0.20, cam_y=0.08)
        self.stop_eps, self.stop_ticks = g('stop_eps'), int(round(g('stop_time') / 0.01))

        # ---- the two stages ------------------------------------------------------------------
        P0 = np.diag([1e-6, 1e-6, 1e-6, g('var_b0'), g('sigma_g') ** 2])
        x0 = [g('init_x'), g('init_y'), g('init_yaw'), g('b0_hat'), 0.0]
        self.map_ekf = Ekf(x0, P0, params)                     # stage 2: with landmarks
        self.odom_ekf = Ekf([0.0, 0.0, 0.0, g('b0_hat'), 0.0], P0.copy(), params)   # stage 1: none

        # ---- landmark map (known association) --------------------------------------------------
        share = get_package_share_directory('a3_description')
        with open(os.path.join(share, 'config', 'landmarks.yaml')) as f:
            self.map = {int(m['id']): (m['x'], m['y']) for m in yaml.safe_load(f)['landmarks']}

        # ---- ROS plumbing ----------------------------------------------------------------------
        self.buf = Buffer()
        self.tfl = TransformListener(self.buf, self)
        self.br = TransformBroadcaster(self)
        self.cam = None                                       # (cx, cy) of base_link -> camera_link
        self.incs, self.imus = {}, {}                         # by stamp (ns), until paired
        self.last_ns = None
        self.still = 0
        self.dets = []                                        # held detections
        self.n_late = 0
        q = QoSProfile(depth=1000)
        self.create_subscription(WheelIncrement, '/wheel_increments', self.on_inc, q)
        self.create_subscription(Imu, '/imu', self.on_imu, q)
        self.create_subscription(LandmarkObservations, '/landmarks', self.on_det, 50)
        self.pub_map = self.create_publisher(Odometry, '/ekf/odom', 10)
        self.pub_own = self.create_publisher(Odometry, '/odom_own', 10)
        self.pub_nis = self.create_publisher(Float64, '/ekf/nis', 10)
        self.pub_b = self.create_publisher(Float64, '/ekf/bias', 10)
        self.pub_b_std = self.create_publisher(Float64, '/ekf/bias_std', 10)
        self.get_logger().info('ekf node up; waiting for /wheel_increments, /imu and the clock')

    # ---- pairing: one tick = one IMU sample + the increment with the same stamp ----------------
    def on_inc(self, m):
        k = stamp_ns(m.header.stamp)
        self.incs[k] = (m.ds, m.dtheta)
        self.pair(k)

    def on_imu(self, m):
        k = stamp_ns(m.header.stamp)
        self.imus[k] = m.angular_velocity.z
        self.pair(k)

    def pair(self, k):
        if k in self.incs and k in self.imus:
            ds, dth = self.incs.pop(k)
            self.tick(k, ds, dth, self.imus.pop(k))
        for d in (self.incs, self.imus):                       
            for old in [o for o in d if o < k - 2 * NS]:
                del d[old]

    # ---- one tick: predict, zero-rate update, publish ----------------------------------------
    def tick(self, k, ds, dth, omega):
        dt = 0.01 if self.last_ns is None else (k - self.last_ns) * 1e-9
        self.last_ns = k
        if dt <= 0.0:
            return
        still = abs(ds) < self.stop_eps and abs(dth) < self.stop_eps
        self.still = self.still + 1 if still else 0
        stopped = self.still >= self.stop_ticks

        for ekf in (self.odom_ekf, self.map_ekf):
            ekf.predict(ds, dth, omega)
            if stopped:
                nis = ekf.update_zero_rate(omega)
                if nis is not None and ekf is self.map_ekf:
                    self.pub_nis.publish(Float64(data=float(nis)))
            else:
                ekf.update_gyro(omega, ds, dth)
        self.publish(k)
        self.drain(k)

    # ---- detections -----------------------------------------------------------------------------
    def on_det(self, m):
        self.dets.append((stamp_ns(m.header.stamp), m.observations))
        if self.last_ns is not None:
            self.drain(self.last_ns)

    def camera_offset(self):
        """base_link -> camera_link from tf2: the camera has no rotation, only an offset."""
        if self.cam is None:
            try:
                t = self.buf.lookup_transform('base_link', 'camera_link', Time(),
                                              timeout=Duration(seconds=0.0))
                self.cam = (t.transform.translation.x, t.transform.translation.y)
                for ekf in (self.odom_ekf, self.map_ekf):
                    ekf.p['cam_x'], ekf.p['cam_y'] = self.cam
            except Exception:                                   # not available yet
                return None
        return self.cam

    def drain(self, now_ns):
     
        cam = self.camera_offset()
        if cam is None:
            return
        keep = []
        for ns, obs in self.dets:
            if ns > now_ns:
                keep.append((ns, obs))
                continue
            if now_ns - ns > 1 * NS:                            # older than 1 s: drop and count
                self.n_late += 1
                continue
            for o in obs:
                if o.id not in self.map:
                    continue
                nis = self.map_ekf.update_landmark(self.map[o.id], o.range, o.bearing)
                if nis is not None:
                    self.pub_nis.publish(Float64(data=float(nis)))
        self.dets = keep

    # ---- outputs --------------------------------------------------------------------------------
    def publish(self, k):
        xo, Po = self.odom_ekf.x, self.odom_ekf.P
        xm, Pm = self.map_ekf.x, self.map_ekf.P
        idx = [0, 1, 2]

        own = Odometry()
        fill_odom(own, 'odom', 'base_link', k, xo, Po[np.ix_(idx, idx)])
        self.pub_own.publish(own)
        est = Odometry()
        fill_odom(est, 'map', 'base_link', k, xm, Pm[np.ix_(idx, idx)])
        self.pub_map.publish(est)
        self.pub_b.publish(Float64(data=float(xm[3] + xm[4])))
        var_b = Pm[3, 3] + Pm[4, 4] + 2.0 * Pm[3, 4]
        self.pub_b_std.publish(Float64(data=float(math.sqrt(max(var_b, 0.0)))))

        # odom -> base_link: the stage-1 pose
        t1 = TransformStamped()
        t1.header = own.header
        t1.child_frame_id = 'base_link'
        t1.transform.translation.x, t1.transform.translation.y = float(xo[0]), float(xo[1])
        t1.transform.rotation = own.pose.pose.orientation
        # map -> odom = T_map<-base * inverse(T_odom<-base), in the plane
        th = xm[2] - xo[2]
        c, s = math.cos(th), math.sin(th)
        t2 = TransformStamped()
        t2.header.stamp, t2.header.frame_id = ns_to_msg(k), 'map'
        t2.child_frame_id = 'odom'
        t2.transform.translation.x = float(xm[0] - (c * xo[0] - s * xo[1]))
        t2.transform.translation.y = float(xm[1] - (s * xo[0] + c * xo[1]))
        t2.transform.rotation.z, t2.transform.rotation.w = math.sin(0.5 * th), math.cos(0.5 * th)
        self.br.sendTransform([t1, t2])


def main():
    rclpy.init()
    node = EkfNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
