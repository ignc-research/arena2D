/* Author: Kyra Kerz
 * 1. Human Wanderer with two shapes on one body
 * 2. Functionality for interaction between two or more human wanderer
 * 3. Simulating movement while destroying and rebuilding body with changing step width between the two shapes
 * 4. Velocity and Angle set randomly after stopping in a predefined range
 */

#include "WandererBipedal.hpp"
#include "math.h"

WandererBipedal::WandererBipedal(b2World * w, const b2Vec2 & position,
					float velocity, float change_rate, float stop_rate, float max_angle_change, unsigned int type):
					Wanderer(w, position, velocity, change_rate, stop_rate, max_angle_change, type)
{
            
	float r = HUMAN_LEG_SIZE/2.f;
	float offset = HUMAN_LEG_DISTANCE/2.0f;
    step_frequency_factor = 0.1;
    step_width_factor = 0.1;
	addCircle(r, b2Vec2(offset+r, 0));
	addCircle(r, b2Vec2(-offset-r, 0));


	//variables for wanderers interacting with each other
    float chat_max_time = _SETTINGS->stage.max_time_chatting / _SETTINGS->physics.time_step / _SETTINGS->physics.step_iterations;
	chat_counter = 0;
	chat_threshold = (int)f_frandomRange(0, chat_max_time);
	chat_reset_counter = 0;
	chat_reset_threshold = (int)chat_max_time/2;
}

void WandererBipedal::update(bool chat_flag)
{

    //check if this wanderer is currently near another wanderer (chat flag == true)
    if(chat_flag){
        //check if interaction time end is reached, if not then stay
        if(chat_counter < chat_threshold){
            chat_counter++;
            _body->SetAngularVelocity(0);
            _body->SetLinearVelocity(b2Vec2_zero);
            return;
        }
    }else{
        //makes sure, the wanderers are able to walk away from each other
        if(chat_counter >= chat_threshold){
            chat_reset_counter++;
            //resets chat_counter in order to allow wanderer interaction again
            if(chat_reset_counter > chat_reset_threshold){
                chat_counter = 0;
            }
        }else{
            chat_reset_counter = 0;
            chat_counter = 0;
        }
    }

    // destroy ald body in order to create a new one
    b2Fixture *fixOld1 = _body->GetFixtureList();
    _body->DestroyFixture(fixOld1);
    fixOld1 = _body->GetFixtureList();
    _body->DestroyFixture(fixOld1);

    //create new circles with changing positions to emulate walking
    float y_val = fmod( _counter,  2*M_PI);
    float legRadius = HUMAN_LEG_SIZE/2.f;
    b2Vec2 circle1Pos = b2Vec2(sin(y_val)*step_width_factor, 0.05);
    addCircle(legRadius, circle1Pos);
    b2Vec2 circle2Pos = b2Vec2(sin(y_val+ M_PI)*step_width_factor, -0.05);
    addCircle(legRadius, circle2Pos);

    if(_counter > 100*M_PI){
        _counter = 0;
    }else{
        _counter = _counter + step_frequency_factor;
    }
//    if(f_random() <= _changeRate)
        updateVelocity();
}


void WandererBipedal::updateVelocity()
{
    //printf("update velocity\n");
    float max_angle = f_rad(30);
    float max_velocity = _velocity;
    b2Vec2 v = _body->GetLinearVelocity();
    float angle;
    zVector2D v_rot = zVector2D(v.x, v.y);
    float lo = max_velocity/4;
    float hi = max_velocity/2;
    float random_velocity = lo + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(hi-lo)));
    //printf("calc velocity\n");
    //wanderer stops for a random rate, in order to change direction and velocity
    if(f_random() < _stopRate){
        //printf("stop!\n");

        v.Set(0,0);
    }
    else{
        //each time wanderer stops, a new random velocity and orientation is set
        if(v == b2Vec2_zero){
            //setting random velocity after each stop
            float sign_x = 1;
            if(rand()%2 == 0) sign_x = -1;
            float sign_y = 1;
            if(rand()%2 == 0) sign_y = -1;

            float lo_y = max_velocity/5;
            float hi_y = max_velocity/3;
            float random_velocity_y = lo_y + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(hi_y-lo_y)));

            v_rot = zVector2D(sign_x* random_velocity, sign_y * random_velocity_y);//.getRotated(angle);
        }
        else{


            angle = (max_angle*2*f_random())-max_angle;
//          v_rot = max_velocity y/2 * zVector2D(v.x, v.y).getNormalized();
            float x = 0;
            float y = 0;

            //preventing that wanderer gets slower
            if (v.x < 0 ) {
                x = -random_velocity;
            } else {
                x = random_velocity;
            }
            y = v.y;

            //change y orientation in order talk walk diagonal

            if(rand()%5 == 0) y * 0.023;
            if(rand()%7 == 0) y * -0.023;

            v_rot = zVector2D(x, y);
        }
        v.Set(v_rot.x, v_rot.y);
    }
    //printf("set velocity\n");
    _body->SetLinearVelocity(b2Vec2(v_rot.x, v_rot.y));
    //wanderer orientation  faces in direction of velocity
    _body->SetTransform(_body->GetPosition(), atan2(v_rot.y,v_rot.x));
    //printf("update DONE\n");

}
