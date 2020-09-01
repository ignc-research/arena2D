#ifndef EVALUATION
#define EVALUATION

#include <list>
#include <iostream>
#include <fstream>
#include <string>
#include <engine/GlobalSettings.hpp>
#include <engine/zVector2d.hpp>
#include "Robot.hpp"

class Evaluation
{
public:
    Evaluation();
    ~Evaluation();

    void init();

    void countHuman();
    void countWall();
    void countGoal();
    void countTimeout();
    void saveDistance(std::list<float> & distances);
    void countAction(const b2Transform & robot_transform);

    void saveData();

private:
    void episodeEnd();

    unsigned int episode_counter;
    unsigned int human_counter;
    unsigned int wall_counter;
    unsigned int goal_counter;
    unsigned int Timeout_counter;
    
    unsigned int action_counter;
    float travelled_distance;
    b2Vec2 _old_robot_position;

    std::ofstream myfile;
};
#endif