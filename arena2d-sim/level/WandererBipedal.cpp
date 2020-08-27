#include "WandererBipedal.hpp"
#include "math.h"

WandererBipedal::WandererBipedal(b2World * w, const b2Vec2 & position,
					float velocity, float change_rate, float stop_rate, float max_angle_change, unsigned int type):
					Wanderer(w, position, velocity, change_rate, stop_rate, max_angle_change, type)
{
	float r = HUMAN_LEG_SIZE/2.f;
	float offset = HUMAN_LEG_DISTANCE/2.0f;
	addCircle(r, b2Vec2(0, offset+r));
	addCircle(r, b2Vec2(0, -offset-r));

	step_frequency_factor = 0.02;
    step_width_factor = 0.1;
}

void WandererBipedal::update(){
	//update foot position
	float y_val = fmod( _counter,  2*M_PI);
	float r = HUMAN_LEG_SIZE/2.f;
	float offset = HUMAN_LEG_DISTANCE/2.0f;

	b2Shape *foot1 = _body->GetFixtureList()->GetShape();
	b2CircleShape *foot1_circle = dynamic_cast<b2CircleShape*>(foot1);
	foot1_circle->m_p.Set(sin(y_val)*step_width_factor, offset+r);

	b2Shape *foot2 = _body->GetFixtureList()->GetNext()->GetShape();
	b2CircleShape *foot2_circle = dynamic_cast<b2CircleShape*>(foot2);
	foot2_circle->m_p.Set(sin(y_val+ M_PI)*step_width_factor, -offset-r);

	if(_counter > 100*M_PI){
        _counter = 0;
    }else{
        _counter = _counter + step_frequency_factor;
    }
	
	updateVelocity();

	/*
	if(f_random() <= _changeRate)
		updateVelocity();
	*/
}

void WandererBipedal::updateVelocity()
{
    float max_angle = f_rad(30);
    float max_velocity = _velocity;
    b2Vec2 v = _body->GetLinearVelocity();
    float angle;
    zVector2D v_rot;

    if(f_random() < _stopRate){
        v.Set(0,0);
    }
    else{
        if(v == b2Vec2_zero){
            angle = 2*M_PI*f_random();
            v_rot = zVector2D(max_velocity/2, 0);//.getRotated(angle);
        }
        else{
            angle = (max_angle*2*f_random())-max_angle;
            v_rot = max_velocity/2 * zVector2D(v.x, v.y).getNormalized();//.getRotated(angle);
        }
    }
    _body->SetLinearVelocity(b2Vec2(v_rot.x, v_rot.y));
    _body->SetTransform(_body->GetPosition(), atan2(v_rot.y,v_rot.x));
}
