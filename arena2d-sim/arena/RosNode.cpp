#include "RosNode.hpp"

RosNode::RosNode(int num_envs, int argc, char **argv) : m_num_envs(num_envs)
{
    m_actions_buffer = std::vector<std::unique_ptr<Twist>>();
    for (int i = 0; i < num_envs; i++){
        m_actions_buffer.emplace_back(new Twist(0, 0));
    }
    m_num_action_msgs_received = 0;
    /* init ros without sigint handler */
    ros::init(argc, argv, "arena_sim", ros::init_options::NoSigintHandler);
    m_nh_ptr = std::unique_ptr<ros::NodeHandle>(new ros::NodeHandle("arena2d"));
    _publishParams();
    _subscribeActions();
}
RosNode::~RosNode(){};

void RosNode::_publishParams()
{
    m_nh_ptr->setParam("settings/num_envs", m_num_envs);
}

void RosNode::_subscribeActions()
{
    for (int i = 0; i < m_num_envs; i++){
        std::stringstream ss;
        ss << "env_" << i << "/action";
        m_action_subscribers.push_back(m_nh_ptr->subscribe<geometry_msgs::Twist>(ss.str(), 1, boost::bind(&RosNode::_ActionMsgCallback, this, _1, i)));
    }
}

void RosNode::_ActionMsgCallback(const geometry_msgs::Twist::ConstPtr & action_msg, int idx_action){
     m_actions_buffer[idx_action]->linear = action_msg->linear.x;
     m_actions_buffer[idx_action]->angular = action_msg->angular.x;
     m_num_action_msgs_received++;
}

bool RosNode::getActions(Twist* robot_Twist,float waitTime=0){
    ros::getGlobalCallbackQueue()->callAvailable(ros::WallDuration(waitTime));
    if(m_num_action_msgs_received == m_num_envs){
        for (int i = 0; i < m_num_envs; i++){
            robot_Twist[i].angular = m_actions_buffer[i]->angular;
            robot_Twist[i].linear = m_actions_buffer[i]->linear;
            m_num_action_msgs_received = 0;
        }
        return true;
    }
    else
    {
        return false;
    }
}

