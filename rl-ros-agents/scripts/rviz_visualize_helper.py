import rospy
from arena2d_msgs.msg import Arena2dResp
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import LaserScan
import argparse
import time
import tf
import tf.transformations as tft
import numpy as np
from collections import deque
import threading


class IntermediateRosNode:
    """
        rviz doesn't support show custom message, we need to either split the
        custom message to standard message or write rviz plug,the former is much
        easier.
    """

    def __init__(self, idx_env=0, laser_scan_publish_rate: int = 0):
        """
        Args:
            idx_env: the environment index we want to visualze by rviz. Defaults to 0
            laser_scan_publish_rate (int, optional): [the pushlish rate to the laser scan, if the it is set to 0
            then the publishment of the laser scan will synchronized with receiving message]. Defaults to 0.
        """
        self._robot_frame_id = "arena_robot_{:02d}".format(idx_env)
        self._header_seq_id = 0
        rospy.init_node("arena_env{:02d}_redirecter".format(idx_env), anonymous=True)
        self._idx_env = idx_env
        if laser_scan_publish_rate == 0:
            # a flag to show where the laser scan is publised in a asynchronized way
            self._is_laser_scan_publish_asyn = False
        else:
            self._is_laser_scan_publish_asyn = True
            # set a cache and lock for accessing it in multi-threading env
            # we set the maxlen to three, since we want to
            # give more info about current status
            self._laser_scan_cache = deque(maxlen=3)
            self._laser_scan_cache_lock = threading.Lock()
            self._laser_scan_pub_rate = laser_scan_publish_rate
            self._laser_scan_pub_rater = rospy.Rate(hz=laser_scan_publish_rate)
            self._new_laser_scan_received = False
            self.start_time = rospy.Time.now()
        self._setSubPub()

    def _setSubPub(self, laser_scan_publish_rate: int = 0):
        namespace_sub = "arena2d/env_{:d}/".format(self._idx_env)
        namespace_pub = "arena2d_intermediate/"

        # publisher
        self._pub_laser_scan = rospy.Publisher(namespace_pub + "laserscan", LaserScan, queue_size=1, tcp_nodelay=True)
        #
        # self. = rospy.Publisher(namespace_pub+"robot_pos", Pose2D, queue_size=1, tcp_nodelay=True)
        # transform broadcaseter for robot position
        # self._tf_rospos = tf.TransformBroadcaster(queue_size=1)
        self._tf_rospos = tf.TransformBroadcaster()
        rospy.loginfo("intermediate node is waiting for connecting env[{:02d}]".format(self._idx_env))
        times = 0
        # subscriber
        # According to the testing,enable tcp_nodelay can double the performance
        self._sub = rospy.Subscriber(namespace_sub + "response", Arena2dResp,
                                     self._arena2dRespCallback, tcp_nodelay=True)
        # # give rospy enough time to establish the connection, without this procedure, the message to
        # # be published at the beginning could be lost.
        while self._sub.get_num_connections() == 0:
            time.sleep(0.1)
            times += 1
        rospy.loginfo("Successfully connected with arena-2d simulator, took {:3.1f}s.".format(.1 * times))

    def _arena2dRespCallback(self, resp: Arena2dResp):
        curr_time = rospy.Time.now()
        robot_pos = resp.robot_pos
        self._tf_rospos.sendTransform((robot_pos.x, robot_pos.y, 0),
                                      tft.quaternion_from_euler(0, 0, robot_pos.theta),
                                      curr_time,
                                      self._robot_frame_id, "world")
        laser_scan = resp.observation
        #
        laser_scan.angle_min = 0
        laser_scan.angle_max = 2 * np.pi
        laser_scan.angle_increment = np.pi/180
        laser_scan.range_min = 0
        # not sure about it.
        laser_scan.range_max = 5

        # set up header
        laser_scan.header.frame_id = self._robot_frame_id
        laser_scan.header.seq = self._header_seq_id
        laser_scan.header.stamp = curr_time

        self._header_seq_id += 1

        if not self._is_laser_scan_publish_asyn:
            self._pub_laser_scan.publish(laser_scan)
        else:
            with self._laser_scan_cache_lock:
                self._laser_scan_cache.append(laser_scan)

    def run(self):
        while not rospy.is_shutdown():
            if self._is_laser_scan_publish_asyn:
                with self._laser_scan_cache_lock:
                    len_cache = len(self._laser_scan_cache)
                    # if no cache, do nothing
                    # if cache size == 2 , laser scan pubish rate too slow compared to coming data
                    if len_cache == 0:
                        continue
                    else:
                        latest_laser_scan = self._laser_scan_cache[-1]
                        self._pub_laser_scan.publish(latest_laser_scan)
                        # it means the the interact rate is too high, some of the data will be discarded
                        if len_cache == 3 and rospy.Time.now().to_sec()-self.start_time.to_sec() > 10:
                            interact_rate = 2*1 / \
                                (self._laser_scan_cache[-1].header.stamp.to_sec() -
                                 self._laser_scan_cache[0].header.stamp.to_sec())
                            rospy.logwarn_once("The rate [{:3.1f} FPS] of republishment of the laser scan is lower compared to the \
                            receivings [approximately {:3.1f} FPS], therefore some the them are discareded".format(self._laser_scan_pub_rate, interact_rate))
                        # chane the cache size to 1
                        while len(self._laser_scan_cache) != 1:
                            self._laser_scan_cache.popleft()
                self._laser_scan_pub_rater.sleep()
            else:
                rospy.spin()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=int, default=0,
                        help="the index of the environment whose response message need to be redicted!")
    parser.add_argument("--laser_scan_pub_rate", type=int, default=5,
                        help="set up the publishing rate of the laser scan, if it is set to 0, then the rate is synchronized with\
                        the receiving rate")
    args = parser.parse_args()
    helper_node = IntermediateRosNode(args.env, args.laser_scan_pub_rate)
    helper_node.run()
