#pragma once
// #include "Robot.hpp"
#include <thread>
#include <chrono>
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <ros/callback_queue.h>
#include <arena/Environment.hpp>
#include <arena2d_msgs/RosAgentReq.h>
#include <arena2d_msgs/Arena2dResp.h>
#include <engine/GlobalSettings.hpp>

class RosNode
{
    using size_t = unsigned int;

private:
    void _publishParams();
    /* callback function for handle ros agent request */
    void _RosAgentReqCallback(const arena2d_msgs::RosAgentReq::ConstPtr  &action_msg, int idx_action);
    void _setRosAgentsReqSub();
    void _setArena2dResPub();


    std::vector<std::unique_ptr<Twist>> m_actions_buffer;
    std::vector<ros::Subscriber> m_ros_agent_subs;
    std::vector<ros::Publisher> m_arena2d_pubs;
    int m_num_envs;
    std::unique_ptr<ros::NodeHandle> m_nh_ptr;
    size_t m_num_ros_agent_req_msgs_received; // request messages are received by different topic name and they will temporary saved in the buffer. only this variable is equal to the number to the envirments, the contents in the buffer can be synchronized with the Buffer in Arena.
    std::vector<bool> m_envs_reset;
    bool m_any_env_reset;
    int m_env_close; // number of envs request to close
    Environment* m_envs;
   

public:
    bool m_env_connected;
    enum class Status
    {
        NOT_ALL_AGENT_MSG_RECEIVED,
        ALL_AGENT_MSG_RECEIVED,
        ENV_RESET,
        SIM_CLOSE,
        BAD_MESSAGE
    };
    RosNode(Environment* envs,int num_envs, int argc, char **argv);
    RosNode(const RosNode&)=delete;
    ~RosNode();
    void publishStates(const bool *dones, float mean_reward = 0, float mean_sucess = 0);

    /*  Synchronize the actions in temporary buffer with the buffer in the class Arena, 
     *  if return false, Synchronization is not done. it means not all the messages are received 
     *  
     */
    Status getActions(Twist *robot_Twist, bool* ros_envs_reset, float waitTime);
    void waitConnection();
};
