#include "Wanderers.hpp"

void Wanderers::init(){
    // adding a human wanderers
    for(int i = 0; i < _SETTINGS->stage.num_dynamic_obstacles; i++){
        _wanderers.push_back(new WandererBipedal(_levelDef.world, b2Vec2(i-1,-1), 
                                                _SETTINGS->stage.obstacle_speed,
                                                0.1, 0.05, 60.0f, WANDERER_ID_HUMAN));
    }

    /*
    // adding a robot wanderer
    for(int i = 0; i < _SETTINGS->stage.num_dynamic_obstacles; i++){
        _wanderers.push_back(new Wanderer(	WANDERER_ROBOT_SIZE,
                                            b2Vec2(0,0),
                                            WANDERER_ROBOT_VELOCITY,
                                            0.1, 0.0,
                                            WANDERER_ID_ROBOT));
    }
    */
}


void Wanderers::freeWanderers(){
    // free all wanderers
    for(int i = 0; i < _wanderers.size(); i++){
        delete(_wanderers[i]);
    }
    _wanderers.clear();
}


void Wanderers::reset(std::vector<b2Vec2> & spawn_position){
    for(int i = 0; i < _wanderers.size(); i++){
        if(i < spawn_position.size()){
            _wanderers[i]->reset(spawn_position[i]);
        }else{
            _wanderers[i]->reset(b2Vec2(i-1, -1));
        }
	}
    //reset all lists
    _old_observed_wanderers.clear();
    _observed_wanderers.clear();
    _infos_of_wanderers.clear();
    _distance_evaluation.clear();
}

void Wanderers::update(){
    // updating all wanderers -> this adjusts the wanderers moving direction
    for(int i = 0; i < _wanderers.size(); i++){
        _wanderers[i]->update();
    }
    calculateDistanceAngle();
    getClosestWanderers();
}

void Wanderers::calculateDistanceAngle(){
    //clear and update all observation vectors of wanderers
    _old_observed_wanderers.clear();
    _old_observed_wanderers = _observed_wanderers;
    _observed_wanderers.clear();
    _infos_of_wanderers.clear();
    _distance_evaluation.clear();

    //calculate distance and angle of all wanderers relativ to robot
    for(int i = 0; i < _wanderers.size(); i++){
        b2Transform robot_transform = _levelDef.robot->getBody()->GetTransform();

        // calculate distance from (_wanderers[i]+safetyDistance) to robot
        b2Vec2 robot_to_wanderer = _wanderers[i]->getPosition() - robot_transform.p;
        float dist = robot_to_wanderer.Length() - _levelDef.robot->getRadius() - _wanderers[i]->getRadius();
        
        //calculate angle of wanderer relativ to the robots facing direction
        zVector2D robot_facing(1, 0);
        robot_facing.rotate(robot_transform.q.GetAngle());// robot facing vector
        float angle = f_deg(zVector2D::signedAngle(robot_facing, zVector2D(robot_to_wanderer.x, robot_to_wanderer.y)));// angle between robot facing vector and robot to wanderer
        
        _infos_of_wanderers.push_back(WandererInfo(i, dist, angle));

        //save distances for evaluation
        _distance_evaluation.push_back(dist);
    }
    _evaluation.saveDistance(_distance_evaluation);
}

void Wanderers::getClosestWanderers(){
    //sort by distances
    _infos_of_wanderers.sort();

    //Get only Wanderers inside the camera view
    float half_camera_angle = _SETTINGS->robot.camera_angle/2.;
    for(std::list<WandererInfo>::iterator it = _infos_of_wanderers.begin(); it != _infos_of_wanderers.end(); it++){
        if(it->angle < half_camera_angle && it->angle > -half_camera_angle){
            _observed_wanderers.push_back(*it);
        }
    }
}

void Wanderers::get_old_Distance(std::vector<float> & old_distance){
    for(int i = 0; i < _old_observed_wanderers.size(); i++){
        old_distance.push_back(_old_observed_wanderers[i].distance);
    }
}

void Wanderers::get_Distance(std::vector<float> & distance){
    for(int i = 0; i < _old_observed_wanderers.size(); i++){
        for(std::list<WandererInfo>::iterator it = _infos_of_wanderers.begin(); it != _infos_of_wanderers.end(); it++){
            if(it->index == _old_observed_wanderers[i].index){
                distance.push_back(it->distance);
            }
        }
    }
}

void Wanderers::getWandererData(std::vector<float> & data){
    for(int i = 0; i < _SETTINGS->training.num_obs_humans; i++){
        if(i < _observed_wanderers.size()){
            data.push_back(_observed_wanderers[i].distance);		// distance to closest
		    data.push_back(_observed_wanderers[i].angle);		// angle to closest (relative from robot)
        }else{
            //Fill with default values
            data.push_back(_SETTINGS->stage.level_size);		// largest distance in level
			data.push_back(0.);		// wanderer is in front of robot
        }
    }
}

bool Wanderers::checkHumanContact(b2Fixture* other_fixture){
    bool humanContact = false;
    for(int i = 0; i < _wanderers.size(); i++){
        if(_wanderers[i]->getType() == WANDERER_ID_HUMAN){
            if(_wanderers[i]->getBody()->GetFixtureList() == other_fixture){
                humanContact = true;
            }
        }
    }
    return humanContact;
}

