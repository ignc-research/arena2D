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

struct WandererInfo {
    int index;
    float distance;
    float angle;

    WandererInfo(int _index, float _distance, float _angle): index(_index), distance(_distance), angle(_angle) {}
    
    bool operator <(const WandererInfo & obj) const {
        return distance < obj.distance;
    }
};

class Wanderers
{
public:
    Wanderers(const LevelDef & levelDef):_levelDef(levelDef){}
    ~Wanderers(){freeWanderers();}

    void freeWanderers();

    void reset(RectSpawn & _dynamicSpawn);

    void update();

    void get_old_observed_distances(std::vector<float> & old_distance);
    void get_observed_distances(std::vector<float> & distance);

    void get_old_distances(std::vector<float> & old_distance);
    void get_distances(std::vector<float> & distance);

    void getWandererData(std::vector<float> & data);
    bool checkHumanContact(b2Fixture* other_fixture);
    
private:
    void calculateDistanceAngle();
    void getClosestWanderers();

    // private variables
    /* level initializer */
	LevelDef _levelDef;

	/* list that stores all wanderers for dynamic level */
    std::vector<WandererBipedal*> _wanderers;

    std::list<WandererInfo> _infos_of_wanderers;
    std::list<WandererInfo> _old_infos_of_wanderers;
    std::vector<WandererInfo> _observed_wanderers;
    std::vector<WandererInfo> _old_observed_wanderers;


    std::list<float> _distance_evaluation;
};

#endif