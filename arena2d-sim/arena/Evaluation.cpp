#include "Evaluation.hpp"

Evaluation::Evaluation(){
    episode_counter = 0;
    human_counter = 0;
    wall_counter = 0;
    goal_counter = 0;
    Timeout_counter = 0;

    action_counter = 0;
    travelled_distance = 0;
}

Evaluation::~Evaluation(){

}

void Evaluation::init(){
    if(_SETTINGS->training.do_evaluation){
        episode_counter = 0;
        human_counter = 0;
        wall_counter = 0;
        goal_counter = 0;
        Timeout_counter = 0;

        action_counter = 0;
        travelled_distance = 0;

        myfile.open ("evaluation.csv");
    }
}

void Evaluation::countHuman(){
    if(_SETTINGS->training.do_evaluation){
        human_counter++;
        episodeEnd();
    }
}

void Evaluation::countWall(){
    if(_SETTINGS->training.do_evaluation){
        wall_counter++;
        episodeEnd();
    }
}

void Evaluation::countGoal(){
    if(_SETTINGS->training.do_evaluation){
        goal_counter++;
        episodeEnd();
    }
}

void Evaluation::countTimeout(){
    if(_SETTINGS->training.do_evaluation){
        Timeout_counter++;
        episodeEnd();
    }
}

void Evaluation::saveDistance(std::list<float> & distances){
    if(_SETTINGS->training.do_evaluation){
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
    if(_SETTINGS->training.do_evaluation){
        action_counter++;

        //calculate travelled distance only if _old_robot_position has usefull information 
        if(action_counter > 1){
            b2Vec2 distance = _old_robot_position - robot_transform.p;
            travelled_distance += distance.Length();
        }
        _old_robot_position = robot_transform.p;
    }
}

void Evaluation::episodeEnd(){
    episode_counter++;

    myfile << "Episode ," << episode_counter << std::endl;
    myfile << "Action counter ," << action_counter << std::endl;
    myfile << "Travelled distance ," << travelled_distance << std::endl;
    myfile << std::endl;

    //reset action counter and traveled distance per Episode
    action_counter = 0;
    travelled_distance = 0;
}

void Evaluation::saveData(){
    if(_SETTINGS->training.do_evaluation){
        myfile << "Goal counter ," << goal_counter << std::endl;
        myfile << "Human counter ," << human_counter << std::endl;
        myfile << "Wall counter ," << wall_counter << std::endl;
        myfile << "Time out counter," << Timeout_counter << std::endl;
        myfile.close();
    }
}