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

    void init(const char * model);

    void reset();
    void countHuman();
    void countWall();
    void countGoal();
    void countTimeout();
    void saveDistance(std::list<float> & distances);
    void countAction(const b2Transform & robot_transform);
    void saveGoalDistance(float goal_distance, float goal_angle);

    void saveData();

private:
    bool initialized = false;

    unsigned int episode_counter;

    std::ofstream myfile;
};
#endif