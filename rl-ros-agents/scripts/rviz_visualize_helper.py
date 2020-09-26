import rospy
from arena2d_msgs.msg import Arena2dResp
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import LaserScan
import argparse
import time


class IntermediateRosNode:
    """
        rviz doesn't support show custom message, we need to either split the 
        custom message to standard message or write rviz plug,the former is much 
        easier. 
    """

    def __init__(self, idx_env):
        """
            idx_env: the environment index we want to visualze by rviz.
        """
        self._frame_id = "arena_robot" 
        rospy.init_node("arena_env{:02d}_redirecter".format(idx_env), anonymous=True)
        self._idx_env = idx_env
        self._setSubPub()
        

    def _setSubPub(self):
        namespace_sub = "arena2d/env_{:d}/".format(self._idx_env)
        namespace_pub = "arena2d_intermediate/"

        # publisher
        self._pub_laser_scan = rospy.Publisher(namespace_pub + "laserscan", LaserScan, queue_size=1, tcp_nodelay=True)
        self._pub_robot_pos = rospy.Publisher(namespace_pub+"robot_pos", Pose2D, queue_size=1, tcp_nodelay=True)
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
        robot_pos = resp.robot_pos
        laser_scan = resp.observation
        laser_scan.header.frame_id = self._frame_id
        self._pub_robot_pos.publish(robot_pos)
        self._pub_laser_scan.publish(laser_scan)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=int, default=0,
                        help="the index of the environment whose response message need to be redicted!")
    args = parser.parse_args()
    helper_node = IntermediateRosNode(args.env)
    helper_node.run()
