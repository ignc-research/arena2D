#include "RosNode.hpp"

RosNode::RosNode(Environment* envs, int num_envs, int argc, char **argv) : m_num_envs(num_envs), m_envs(envs)
{
    m_num_ros_agent_req_msgs_received = 0;
    m_num_env_reset = 0;

    m_actions_buffer = std::vector<std::unique_ptr<Twist>>();
    for (int i = 0; i < num_envs; i++)
    {
        m_actions_buffer.emplace_back(new Twist(0, 0));
    }

    /* init ros without sigint handler */
    ros::init(argc, argv, "arena_sim", ros::init_options::NoSigintHandler);
    m_nh_ptr = std::unique_ptr<ros::NodeHandle>(new ros::NodeHandle("arena2d"));
    _publishParams();
    _setRosAgentsReqSub();
    _setArena2dResPub();
}
RosNode::~RosNode(){};

void RosNode::_publishParams()
{
    m_nh_ptr->setParam("settings/num_envs", m_num_envs);
}

void RosNode::_setRosAgentsReqSub()
{
    for (int i = 0; i < m_num_envs; i++)
    {
        std::stringstream ss;
        ss << "env_" << i << "/request";
        m_ros_agent_subs.push_back(m_nh_ptr->subscribe<arena2d_msgs::RosAgentReq>(ss.str(), 1, boost::bind(&RosNode::_RosAgentReqCallback, this, _1, i)));
    }
}

void RosNode::_setArena2dResPub()
{
    for (int i = 0; i < m_num_envs; i++)
    {
        std::stringstream ss;
        ss << "env_" << i << "/response";
        m_arena2d_pubs.push_back(m_nh_ptr->advertise<arena2d_msgs::Arena2dResp>(ss.str(), 1));
    }
}

void RosNode::publishStates(const bool* dones , float mean_reward , float mean_sucess)
{
    arena2d_msgs::Arena2dResp resp;
    for (int idx_env = 0; idx_env < m_num_envs; idx_env++)
    {
        // set laser scan data
        int num_sample;
        auto laser_data = m_envs[idx_env].getScan(num_sample);
        for (size_t i = 0; i < num_sample; i++)
        {
            resp.observation.ranges.push_back(laser_data[i]);
        }

        arena2d_msgs::Arena2dResp resp;
        
        // set goal position
        float angle = 0, distance = 0;
        m_envs[idx_env].getGoalDistance(distance, angle);
        resp.goal_pos[0] = static_cast<double>(distance);
        resp.goal_pos[1] = static_cast<double>(angle);
        //TODO Whats is additional data? add it and change the msg type if needed

        resp.reward = m_envs[idx_env].getReward();
        resp.done = dones[idx_env];
        resp.mean_reward = mean_reward;
        resp.mean_success = mean_sucess;

        m_arena2d_pubs[idx_env].publish(resp);
    }
    ROS_DEBUG("published states");
}

void RosNode::_RosAgentReqCallback(const arena2d_msgs::RosAgentReq::ConstPtr &req_msg, int idx_action)
{
    if (req_msg->env_reset == 1)
    {
        m_num_env_reset++;
        ROS_DEBUG_STREAM("env " << idx_action << " request reset");
    }
    else
    {
        m_actions_buffer[idx_action]->linear = req_msg->action.linear;
        m_actions_buffer[idx_action]->angular = req_msg->action.angular;
        ROS_DEBUG_STREAM("env " << idx_action << " message received");
    }
    m_num_ros_agent_req_msgs_received++;
}

RosNode::Status RosNode::getActions(Twist *robot_Twist, float waitTime = 0)
{
    ros::getGlobalCallbackQueue()->callAvailable(ros::WallDuration(waitTime));
    if (m_num_ros_agent_req_msgs_received == m_num_envs) // received all msg from all agents.
    {
        if (m_num_env_reset == 0) // all env are normal
        {
            for (int i = 0; i < m_num_envs; i++)
            {
                robot_Twist[i].angular = m_actions_buffer[i]->angular;
                robot_Twist[i].linear = m_actions_buffer[i]->linear;
                m_num_ros_agent_req_msgs_received = 0;
            }
            ROS_DEBUG("Requests from all env received!");
            return RosNode::Status::ALL_AGENT_MSG_RECEIVED;
        }
        else if (m_num_env_reset == m_num_ros_agent_req_msgs_received)
        { // all env request reset
            m_num_ros_agent_req_msgs_received = 0;
            m_num_env_reset = 0;
            ROS_DEBUG("All agents request to reset the environments");
            return RosNode::Status::ALL_ENV_RESET;
        }
        else
        { // not all env request reset, that shouldnt happen
            m_num_ros_agent_req_msgs_received = 0;
            m_num_env_reset = 0;
            return RosNode::Status::BAD_MESSAGE;
        }
    }
    else
    {
        return RosNode::Status::NOT_ALL_AGENT_MSG_RECEIVED;
    }
}
