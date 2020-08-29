#include "WandererBipedal.hpp"
#include "math.h"

WandererBipedal::WandererBipedal(b2World * w, const b2Vec2 & position,
					float velocity, float change_rate, float stop_rate, float max_angle_change, unsigned int type):
					Wanderer(w, position, velocity, change_rate, stop_rate, max_angle_change, type)
{
	float r = HUMAN_LEG_SIZE/2.f;
	float offset = HUMAN_LEG_DISTANCE/2.0f;
    step_frequency_factor = 0.02;
    step_width_factor = 0.1;
	addCircle(r, b2Vec2(offset+r, 0));
	addCircle(r, b2Vec2(-offset-r, 0));
}

void WandererBipedal::update()
{
    // Update Leg posture
    // b2Vec2 v = _body1->GetLinearVelocity();
    // step_width_factor = v.Normalize();

    b2Fixture *fixOld1 = _body->GetFixtureList();
    _body->DestroyFixture(fixOld1);
    fixOld1 = _body->GetFixtureList();
    _body->DestroyFixture(fixOld1);

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
    if(f_random() <= _changeRate)
        updateVelocity();
}

void WandererBipedal::updateVelocity()
{
    printf("update velocity\n");
    float max_angle = f_rad(30);
    float max_velocity = _velocity;
    b2Vec2 v = _body->GetLinearVelocity();
    float angle;
    zVector2D v_rot = zVector2D(v.x, v.y);

    printf("calc velocity\n");
    if(f_random() < _stopRate){
        printf("stop!\n");

        v.Set(0,0);
    }
    else{
        if(v == b2Vec2_zero){
            printf("is b2Vec2 zero\n");

            float sign = 1;
            if(rand()%2 == 0) sign = -1;

            v_rot = zVector2D(sign* (max_velocity/2), 0);//.getRotated(angle);
        }
        else{
            printf("set velocity else\n");

            angle = (max_angle*2*f_random())-max_angle;
//            v_rot = max_velocity y/2 * zVector2D(v.x, v.y).getNormalized();
            float x = 0;
            float y = 0;
            if (v.x < 0 ) {
                x = -max_velocity/2;
            } else {
                x = max_velocity/2;
            }
            y = v.y;

            if(rand()%5 == 0) y * 0.023;
            if(rand()%7 == 0) y * -0.023;

            v_rot = zVector2D(x, y);
        }
        v.Set(v_rot.x, v_rot.y);
    }
    printf("set velocity\n");
    _body->SetLinearVelocity(b2Vec2(v_rot.x, v_rot.y));
    _body->SetTransform(_body->GetPosition(), atan2(v_rot.y,v_rot.x));
    printf("update DONE\n");
}