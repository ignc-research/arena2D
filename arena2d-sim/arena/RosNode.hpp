#pragma once
#include "Robot.hpp"
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <ros/callback_queue.h>
class RosNode
{
    using size_t = unsigned int;

private:
    void _publishParams();
    /* callback function for handl action message */
    void _ActionMsgCallback(const geometry_msgs::Twist::ConstPtr &action_msg, int idx_action);
    void _subscribeActions();

    std::vector<std::unique_ptr<Twist>> m_actions_buffer;
    std::vector<ros::Subscriber> m_action_subscribers;
    int m_num_envs;
    std::unique_ptr<ros::NodeHandle> m_nh_ptr;
    size_t m_num_action_msgs_received; // actions are received by different topic name and they will temporary saved in the buffer. only this variable is equal to the number to the envirments, the contents in the buffer can be synchronized with the Buffer in Arena.


public:
    RosNode(int num_envs, int argc, char **argv);
    RosNode(const RosNode&)=delete;
    ~RosNode();
    void publishStates();

    /*  Synchronize the actions in the buffer with the buffer in the class Arena, 
     *  if return false, Synchronization is failed. it means not all the messages are received 
     *  
     */
    bool getActions(Twist *robot_twist, float waitTime);
};
