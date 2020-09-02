#include "RosNode.hpp"

RosNode::RosNode(Environment* envs, int num_envs, int argc, char **argv) : m_num_envs(num_envs), m_envs(envs)
{
    ROS_INFO("Ros service activated, waiting for connections from agents");
    m_num_ros_agent_req_msgs_received = 0;
    m_env_close = 0;
    m_any_env_reset = false;
    m_envs_reset.resize(num_envs);

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
    while(1){
        int n_connected=0;
        for(auto& sub: m_ros_agent_subs){
            if(sub.getNumPublishers()!=0){
                n_connected++;
            }
        }
        for(auto& pub: m_arena2d_pubs){
            if(pub.getNumSubscribers()!=0){
                n_connected++;
            }
        }
        if(n_connected != m_arena2d_pubs.size()+m_ros_agent_subs.size()){
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }else{
            break;
        }
    }
    ROS_INFO("All agents successfully connected!");
}
RosNode::~RosNode(){};

void RosNode::_publishParams()
{
    // namespace
    string ns = "~settings";
    ros::NodeHandle nh(ns);
    nh.setParam("num_envs", m_num_envs);
    nh.setParam("action_space_lower_limit", vector<float>{_SETTINGS->robot.backward_speed.linear, _SETTINGS->robot.strong_right_speed.angular});
    nh.setParam("action_space_upper_limit", vector<float>{_SETTINGS->robot.forward_speed.linear, _SETTINGS->robot.strong_left_speed.angular});
    nh.setParam("observation_space_num_beam", _SETTINGS->robot.laser_num_samples);
    nh.setParam("observation_space_upper_limit",_SETTINGS->robot.laser_max_distance);
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
    
    for (int idx_env = 0; idx_env < m_num_envs; idx_env++)
    {   
        // if any env need reset and not current one just skip.
        if(m_any_env_reset && !m_envs_reset[idx_env]){
            continue;
        }
        arena2d_msgs::Arena2dResp resp;
        // set laser scan data
        int num_sample;
        auto laser_data = m_envs[idx_env].getScan(num_sample);
        for (size_t i = 0; i < num_sample; i++)
        {
            resp.observation.ranges.push_back(laser_data[i]);
        }
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
    if(! m_any_env_reset){
        ROS_DEBUG("published states");
    }
    else
    {
        m_any_env_reset = false;
        for (int i = 0; i < m_envs_reset.size();i++)
        {
            m_envs_reset[i] = false;
        }
        ROS_DEBUG("env already reset");
    }
    m_num_ros_agent_req_msgs_received = 0;
}

void RosNode::_RosAgentReqCallback(const arena2d_msgs::RosAgentReq::ConstPtr &req_msg, int idx_env)
{
    if(req_msg->arena2d_sim_close){
        if(m_env_close == 0){
            m_num_ros_agent_req_msgs_received = 0; // reset the counter
        }
        m_env_close += 1;
    }
    if (req_msg->env_reset == 1)
    {
        m_envs_reset[idx_env] = true;
        m_any_env_reset = true;
        ROS_DEBUG_STREAM("env " << idx_env << " request reset");
    }
    else
    {
        m_actions_buffer[idx_env]->linear = req_msg->action.linear;
        m_actions_buffer[idx_env]->angular = req_msg->action.angular;
        ROS_DEBUG_STREAM("env " << idx_env << " message received");
    }
    m_num_ros_agent_req_msgs_received++;
    
}

RosNode::Status RosNode::getActions(Twist *robot_Twist, bool* ros_envs_reset,float waitTime = 0)
{
    ros::getGlobalCallbackQueue()->callAvailable(ros::WallDuration(waitTime));
    if(m_any_env_reset){
        for (int i = 0; i < m_num_envs; i++){
            ros_envs_reset[i] = m_envs_reset[i];
        }
        return RosNode::Status::ENV_RESET;
    }
    if(m_num_ros_agent_req_msgs_received == m_num_envs){
        if (m_env_close == 0) // all env are normal
        {
            for (int i = 0; i < m_num_envs; i++)
            {   
                robot_Twist[i].angular = m_actions_buffer[i]->angular;
                robot_Twist[i].linear = m_actions_buffer[i]->linear;
                
            }
            ROS_DEBUG("Requests from all env received!");
            return RosNode::Status::ALL_AGENT_MSG_RECEIVED;

        }else if(m_env_close == m_num_envs){
            return RosNode::Status::SIM_CLOSE; // only all env wrapper requiring close is considered legal 
        }else{
            return RosNode::Status::BAD_MESSAGE;
        }
    }else{
        return RosNode::Status::NOT_ALL_AGENT_MSG_RECEIVED;
    }
}
