#include "Wanderers.hpp"

void Wanderers::freeWanderers(){
    // free all wanderers
    for(int i = 0; i < _wanderers.size(); i++){
        delete(_wanderers[i]);
    }
    _wanderers.clear();
}

void Wanderers::freeRobotWanderers(){
    // free all wanderers
    for(int i = 0; i < _robot_wanderers.size(); i++){
        delete(_robot_wanderers[i]);
    }
    _robot_wanderers.clear();
}

void Wanderers::reset(RectSpawn & _dynamicSpawn, bool _dynamic, bool _human) {
    if (_wanderers.size() > 0) freeWanderers();
    if (_robot_wanderers.size() > 0) freeRobotWanderers();
    // adding a human wanderers

    if (_dynamic) {
        for (int i = 0; i < _SETTINGS->stage.num_dynamic_obstacles; i++) {
            b2Vec2 p;
            _dynamicSpawn.getRandomPoint(p);

            b2Vec2 p2;
            _dynamicSpawn.getRandomPoint(p2);
            p2.Set(p.x-2,p.y-2);

            b2Vec2 p3;
            _dynamicSpawn.getRandomPoint(p3);
            p3.Set(p2.x+2,p2.y);

            std::vector<b2Vec2> waypoints={p,p2};
            //Wanderer(w, position, velocity,type,mode,waypoints,stop_counter_threshold, change_rate, stop_rate, max_angle_velo)
            
            Wanderer *w =new Wanderer(_levelDef.world, p,_SETTINGS->stage.obstacle_speed,WANDERER_ID_ROBOT,1,
                    waypoints);
            /*
            Wanderer *w = new Wanderer(_levelDef.world, p,
                                       _SETTINGS->stage.obstacle_speed,
                                       0.1, 0.05, 60.0f, WANDERER_ID_ROBOT);
            */
            w->addCircle(_SETTINGS->stage.dynamic_obstacle_size / 2.f);
            _robot_wanderers.push_back(w);
        }
    }
    if(_human){
        for(int i = 0; i < _SETTINGS->stage.num_dynamic_obstacles; i++) {
            b2Vec2 p;
            _dynamicSpawn.getRandomPoint(p);

            b2Vec2 p2;
            _dynamicSpawn.getRandomPoint(p2);
            p2.Set(p.x+1,p.y+1);

            b2Vec2 p3;
            _dynamicSpawn.getRandomPoint(p3);
            p3.Set(p2.x+2,p2.y);

            std::vector<b2Vec2> waypoints={p,p2,p3};

            WandererBipedal *w=new WandererBipedal(_levelDef.world, p,_SETTINGS->stage.obstacle_speed,WANDERER_ID_HUMAN,1,
                    waypoints);

            _wanderers.push_back(w);

            /*
            _wanderers.push_back(new WandererBipedal(_levelDef.world, p,
                                                     _SETTINGS->stage.obstacle_speed,
                                                     0.1, 0.05, 60.0f, WANDERER_ID_HUMAN));
            */
        }
    }
    //reset all lists
    _old_infos_of_wanderers.clear();
    _old_observed_wanderers.clear();
    _infos_of_wanderers.clear();
    _observed_wanderers.clear();
    _distance_evaluation.clear();

    calculateDistanceAngle();
    getClosestWanderers();
}

void Wanderers::update(){
    updateHuman();
    updateRobot();


}

void Wanderers::updateHuman(){
    b2Vec2 position;
    for(int i = 0; i < _wanderers.size(); i++){
        //check if wanderers are near each other -> stop both wanderers as they start chatting
        bool chat_flag = false;
        float radius_check = 0.3;
        for(int j = 0; j < _wanderers.size(); j++){
            if(i == j){
                continue;
            }
            float distance = (_wanderers[i]->getPosition() - _wanderers[j]->getPosition()).Length();
            //printf("Radius");

            if ( distance < radius_check) {
                chat_flag = true;
                break;
            }
        }

        //check if wanderer is out of border
        position = _wanderers[i]->getPosition();

        b2Vec2 new_position = position;
        float border = _SETTINGS->stage.level_size / 2.f;
        float radius = _wanderers[i]->getRadius();
        float bound = border - radius;
        if(abs(position.x) > bound || abs(position.y) > bound ) {
            if((position.x) > bound) {
                new_position.x = bound;
            }
            if((position.x) < -bound){
                new_position.x = -bound;
            }
            if((position.y) > bound){
                new_position.y = bound;
            }
            if((position.y) < -bound){
                new_position.y = -bound;
            }
            _wanderers[i]->setPosition(new_position);
        }
        //update wanderer position and velocity
        _wanderers[i]->update(chat_flag);
    }

    calculateDistanceAngle();
    getClosestWanderers();

}

void Wanderers::updateRobot(){
    for(int i = 0; i < _robot_wanderers.size(); i++){
        _robot_wanderers[i]->update(false);
    }
}


void Wanderers::calculateDistanceAngle(){
    //clear and update all observation vectors of wanderers
    _old_infos_of_wanderers.clear();
    _old_observed_wanderers.clear();
    _old_infos_of_wanderers = _infos_of_wanderers;
    _old_observed_wanderers = _observed_wanderers;
    _infos_of_wanderers.clear();
    _observed_wanderers.clear();
    _distance_evaluation.clear();

    //calculate distance and angle of all wanderers relativ to robot
    for(int i = 0; i < _wanderers.size(); i++){
        b2Transform robot_transform = _levelDef.robot->getBody()->GetTransform();

        // calculate distance from (_wanderers[i]+safetyDistance) to robot
        b2Vec2 robot_to_wanderer = _wanderers[i]->getPosition() - robot_transform.p;
        float dist = robot_to_wanderer.Length() - _levelDef.robot->getRadius() - _wanderers[i]->getRadius();
        
        //calculate angle of wanderer relativ to the robots facing direction
        zVector2D robot_facing(0, 1);
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
    int i = 0;
    for(std::list<WandererInfo>::iterator it = _infos_of_wanderers.begin(); it != _infos_of_wanderers.end(); it++){
        if(i < _SETTINGS->training.num_obs_humans){
            if(it->angle < half_camera_angle && it->angle > -half_camera_angle){
                _observed_wanderers.push_back(*it);
                i++;
            }
        }
    }
}

void Wanderers::get_old_observed_distances(std::vector<float> & old_distance){
    for(int i = 0; i < _old_observed_wanderers.size(); i++){
        old_distance.push_back(_old_observed_wanderers[i].distance);
    }
}

void Wanderers::get_observed_distances(std::vector<float> & distance){
    for(int i = 0; i < _old_observed_wanderers.size(); i++){
        for(std::list<WandererInfo>::iterator it = _infos_of_wanderers.begin(); it != _infos_of_wanderers.end(); it++){
            if(it->index == _old_observed_wanderers[i].index){
                distance.push_back(it->distance);
            }
        }
    }
}

void Wanderers::get_old_distances(std::vector<float> & old_distance){
    for(std::list<WandererInfo>::iterator it_old = _old_infos_of_wanderers.begin(); it_old != _old_infos_of_wanderers.end(); it_old++){
        old_distance.push_back(it_old->distance);
    }
}

void Wanderers::get_distances(std::vector<float> & distance){
    for(std::list<WandererInfo>::iterator it_old = _old_infos_of_wanderers.begin(); it_old != _old_infos_of_wanderers.end(); it_old++){
        for(std::list<WandererInfo>::iterator it = _infos_of_wanderers.begin(); it != _infos_of_wanderers.end(); it++){
            if(it->index == it_old->index){
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
            data.push_back(2*_SETTINGS->stage.level_size);		// largest distance in level
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

