#include "Evaluation.hpp"

Evaluation::Evaluation(){
    episode_counter = 0;
}

Evaluation::~Evaluation(){

}

void Evaluation::init(const char * model){
    char path[200];
    if(model != NULL){
        char * path1;
        char * path2;
        char * model_path = strdup(model);
        path1 = strtok (model_path,"/");
        int i = 0;
        while (path1 != NULL)
        {
            path2 = strtok (NULL, "/");
            if(path2 != NULL){
                if(i == 0) strcpy (path,path1);
                else{
                    strcat (path,"/");
                    strcat (path,path1);
                }
            }   
            path1 = path2;
            i++;
        }
        strcat (path,"/evaluation.csv");
    }else{
        strcpy(path, "evaluation.csv");
    }
    
    printf ("Path to evaluation.csv: %s\n",path);

    episode_counter = 0;

    myfile.open (path);

    myfile << "Episode,Ending,Robot_Position_x,Robot_Position_y,";
    for(int i = 0; i < _SETTINGS->stage.num_dynamic_obstacles; i++){
        myfile << "Human" << i+1;
        if(i < _SETTINGS->stage.num_dynamic_obstacles - 1) myfile << ",";
    }
    myfile << std::endl;

    initialized = true;
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
        myfile << ",,,,";
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
        myfile << ",," << robot_transform.p.x << "," << robot_transform.p.y << std::endl;
    }
}

void Evaluation::saveData(){
    if(initialized){
        myfile.close();
    }
    initialized = false;
}