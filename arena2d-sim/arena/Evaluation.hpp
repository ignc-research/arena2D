#ifndef EVALUATION
#define EVALUATION

#include <list>
#include <iostream>
#include <fstream>
#include <string>
#include <cstddef>
#include <engine/GlobalSettings.hpp>
#include <engine/zVector2d.hpp>
#include "Robot.hpp"

class Evaluation
{
public:
    Evaluation();
    ~Evaluation();

    /* open csv-file inside the training folder, where the model weights are saved
	 * @param model path to trained agent weights
	 */
    void init(const char * model);

    /* when episode is reset -> write "reset" in csv-file
	 */
    void reset();

    /* when human is hit -> write "human" in csv-file
	 */
    void countHuman();

    /* when wall is hit -> write "wall" in csv-file
	 */
    void countWall();

    /* when goal is hit -> write "goal" in csv-file
	 */    
    void countGoal();

    /* when time out -> write "time" in csv-file
	 */
    void countTimeout();

    /* write distances of human wanderers in csv-file
	 * @param distances between robot and human wanderers
	 */
    void saveDistance(std::list<float> & distances);

    /* write robot position and angle after each action in csv-file
	 * @param robot_transform robot position and angle
	 */
    void countAction(const b2Transform & robot_transform);

    /* write goal distance and angle in csv-file, after episode is reset
	 * @param goal_distance distance from robot to goal
     * @param goal_angle goal angle viewed from robot 
	 */
    void saveGoalDistance(float goal_distance, float goal_angle);

    /* close csv-file
	 */
    void saveData();

private:
    /* check if csv-file is opened */
    bool initialized;

    /* count the number of episodes */
    unsigned int episode_counter;

    /* csv-file */
    std::ofstream myfile;
};
#endif