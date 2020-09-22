#ifndef WANDERERS
#define WANDERERS

#include <iostream>
#include <vector>
#include <list>
#include <arena/Evaluation.hpp>

#include "Level.hpp"
#include "Wanderer.hpp"
#include "WandererBipedal.hpp"

extern Evaluation _evaluation;


#define WANDERER_ID_HUMAN 0
#define WANDERER_ID_ROBOT 1

/* WandererInfo to save informations of wanderers and sort them based on their distances
 * @param index to distinguish the wanderers
 * @param distance distance between robot and wanderer
 * @param angle angle of wanderer viewed from robot
 */
struct WandererInfo {
    int index;
    float distance;
    float angle;

    WandererInfo(int _index, float _distance, float _angle): index(_index), distance(_distance), angle(_angle) {}
    
    /* to sort WandererInfo based on the distances */
    bool operator <(const WandererInfo & obj) const {
        return distance < obj.distance;
    }
};

/* The Wanderers class handels all the methods of a level regarding the human wanderers.
 */
class Wanderers
{
public:
    /* constructor
	 * @param levelDef initializer for level containing b2World and Robot
	 */
    Wanderers(const LevelDef & levelDef):_levelDef(levelDef){}

    /* destructor, delete all wanderers */
    ~Wanderers(){freeWanderers();}

    /* free space, delete all wanderers */
    void freeWanderers();

    /* reset wanderers to new random positions 
	 * @param _dynamicSpawn RectSpawn to get random spawn positions for wanderers
	 */
    void reset(RectSpawn & _dynamicSpawn);

    /* update position and velocity of wanderers, while checking if they are chatting or outside the border.
	 * execute calculateDistanceAngle() and getClosestWanderers()
	 */
    void update();

    /* get the old distances of all wanderers, which were inside the camera_view of the robot before the last action was executed
	 * @param old_distance push old observed distances of wanderers into this vector
	 */
    void get_old_observed_distances(std::vector<float> & old_distance);

    /* get the current distances of all wanderers, which were inside the camera_view of the robot before the last action was executed
	 * @param distance push current observed distances of wanderers into this vector
	 */
    void get_observed_distances(std::vector<float> & distance);

    /* get the old distances of all wanderers before the last action was executed
	 * @param old_distance push old distances of wanderers into this vector
	 */
    void get_old_distances(std::vector<float> & old_distance);

    /* get the current distances of all wanderers
	 * @param distance push current distances of wanderers into this vector
	 */
    void get_distances(std::vector<float> & distance);

    /* provide the agent with additional data of wanderers
	 * @param data push distances and angle of wanderers inside the camera_view of the robot and
     * fill it with default values, if the number of observed wanderers is smaller than the expected input of the agent 
	 */
    void getWandererData(std::vector<float> & data);

    /* check if robot had contact with a human
 	 * @return true if contact with human and false otherwise
	 */
    bool checkHumanContact(b2Fixture* other_fixture);
    
private:
    /* calculate distance and angle of all wanderers relativ to the robot
	 */
    void calculateDistanceAngle();

    /* sort wanderers in ascending order according to their distance to the robot
	 * push the wanderers inside the camera view into the _observed_wanderers vector
	 */
    void getClosestWanderers();

    // private variables

    /* level initializer */
	LevelDef _levelDef;

	/* list that stores all wanderers for dynamic level */
    std::vector<WandererBipedal*> _wanderers;

    /* save current wanderer information, used for sorting according to distance */
    std::list<WandererInfo> _infos_of_wanderers;

    /* save old wanderer information, befor last robot action */
    std::list<WandererInfo> _old_infos_of_wanderers;

    /* save current wanderer information inside the camera_view */
    std::vector<WandererInfo> _observed_wanderers;

    /* save old wanderer information inside the camera_view */
    std::vector<WandererInfo> _old_observed_wanderers;

    /* save distances of wanderers for the evaluation */
    std::list<float> _distance_evaluation;
};

#endif