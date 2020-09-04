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
    human_counter = 0;
    wall_counter = 0;
    goal_counter = 0;
    Timeout_counter = 0;

    action_counter = 0;
    travelled_distance = 0;

    myfile.open (path);

    initialized = true;
}

void Evaluation::countHuman(){
    if(initialized){
        human_counter++;
        episodeEnd();
    }
}

void Evaluation::countWall(){
    if(initialized){
        wall_counter++;
        episodeEnd();
    }
}

void Evaluation::countGoal(){
    if(initialized){
        goal_counter++;
        episodeEnd(true);
    }
}

void Evaluation::countTimeout(){
    if(initialized){
        Timeout_counter++;
        episodeEnd();
    }
}

void Evaluation::saveDistance(std::list<float> & distances){
    if(initialized){
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
        action_counter++;

        //calculate travelled distance only if _old_robot_position has usefull information 
        if(action_counter > 1){
            b2Vec2 distance = _old_robot_position - robot_transform.p;
            travelled_distance += distance.Length();
        }
        _old_robot_position = robot_transform.p;
    }
}

void Evaluation::episodeEnd(bool goal_reached){
    episode_counter++;

    myfile << "Episode ," << episode_counter << std::endl;
    if(goal_reached){
        myfile << "Action counter ," << action_counter << std::endl;
        myfile << "Travelled distance ," << travelled_distance << std::endl;
    }
    myfile << std::endl;

    //reset action counter and traveled distance per Episode
    action_counter = 0;
    travelled_distance = 0;
}

void Evaluation::saveData(){
    if(initialized){
        myfile << "Goal counter ," << goal_counter << std::endl;
        myfile << "Human counter ," << human_counter << std::endl;
        myfile << "Wall counter ," << wall_counter << std::endl;
        myfile << "Time out counter," << Timeout_counter << std::endl;
        myfile.close();
    }
    initialized = false;
}