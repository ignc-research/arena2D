#include "Evaluation.hpp"

Evaluation::Evaluation(){
    episode_counter = 0;
    initialized = false;
}

Evaluation::~Evaluation(){

}

void Evaluation::init(const char * model){
    episode_counter = 0;
    initialized = false;

    // get the path to the trainings folder
    std::string str(model);
    std::string path;
    if(model != NULL){
        std::size_t found = str.find_last_of("/\\");
        path = str.substr(0,found) + "/evaluation.csv";
    }else{
        //path to current default location
        path = "evaluation.csv";
    }
    
    std::cout << "Path to evaluation.csv: " << path << std::endl;

    myfile.open(path);

    if(myfile.is_open()){
        // write head of csv file
        myfile << "Episode,Ending,Goal_Distance,Goal_Angle,Robot_Position_x,Robot_Position_y,Robot_Direction_x,Robot_Direction_y,Robot_Action,";
        for(int i = 0; i < _SETTINGS->stage.num_dynamic_obstacles; i++){
            myfile << "Human" << i+1;
            if(i < _SETTINGS->stage.num_dynamic_obstacles - 1) myfile << ",";
        }
        myfile << std::endl;

        initialized = true;
    }else{
        std::cout << "ERROR: Failed to open file evaluation.csv !" << std::endl;
    }
}

void Evaluation::reset(){
    if(initialized){
        myfile << ",reset" << std::endl;
    } 
}

void Evaluation::countHuman(){
    if(initialized){
        myfile << ++episode_counter << ",human" << std::endl;
    }
}

void Evaluation::countWall(){
    if(initialized){
        myfile << ++episode_counter << ",wall" << std::endl;
    }
}

void Evaluation::countGoal(){
    if(initialized){
        myfile << ++episode_counter << ",goal" << std::endl;
    }
}

void Evaluation::countTimeout(){
    if(initialized){
        myfile << ++episode_counter << ",time" << std::endl;
    }
}

void Evaluation::saveDistance(std::list<float> & distances){
    if(initialized){
        myfile << ",,,,,,,,,";
        for(std::list<float>::iterator hd = distances.begin(); hd != distances.end(); hd++){
            if(std::next(hd) == distances.end()){
                myfile << *hd;
            }
            else myfile << *hd << ",";
            
        }
        myfile << std::endl;
    }
}

void Evaluation::countAction(const b2Transform & robot_transform){
    if(initialized){
        // robot facing vector
        zVector2D robot_facing(1, 0);
        robot_facing.rotate(robot_transform.q.GetAngle());
        myfile << ",,,," << robot_transform.p.x << "," << robot_transform.p.y << "," 
        << robot_facing.x << "," << robot_facing.y <<std::endl;
    }
}

void Evaluation::saveAction(Robot::Action a){
    if(initialized){
        myfile << ",,,,,,,," << a <<std::endl;
    }
}

void Evaluation::saveGoalDistance(float goal_distance, float goal_angle){
    if(initialized){
        myfile << ",," << goal_distance << "," << goal_angle << std::endl;
    }
}

void Evaluation::saveData(){
    if(initialized){
        myfile.close();
    }
    initialized = false;
}
