#include "WandererBipedal.hpp"
#include "math.h"

WandererBipedal::WandererBipedal(b2World * w, float radius, const b2Vec2 & position,
					float velocity, float change_rate, float stop_rate, unsigned int type)
{
    printf("Wanderer: init values \n");
	_velocity = velocity;
	_changeRate = change_rate;
	_stopRate = stop_rate;
	_type = type;
	_radius = radius;

    printf("Wanderer create body \n");
	// creating body
	b2BodyDef body_def1;
	body_def1.type = b2_dynamicBody;
	body_def1.allowSleep = false;
    b2Vec2 position1 = position;
	body_def1.position = position1;
	body_def1.linearDamping = 0;
	body_def1.angularDamping = 0;
	body_def1.angularVelocity = 0;
	body_def1.fixedRotation = false;
	body_def1.angle = 0;
	body_def1.userData = (void*)this;
	_body1 = w->CreateBody(&body_def1);

//
//    b2BodyDef body_def2;
//    body_def2.type = b2_dynamicBody;
//    body_def2.allowSleep = false;
//    b2Vec2 position2 = position;
//    position2.x -= 0.1;
//    body_def2.position = position2;
//    body_def2.linearDamping = 0;
//    body_def2.angularDamping = 0;
//    body_def2.userData = (void*)this;
//	_body2 = w->CreateBody(&body_def2); ;


//    b2CircleShape circle2;
//    circle2.m_radius = radius;
//    circle2.m_p.Set(radius+0.025,0);
//    b2FixtureDef d2;
//    d2.filter.categoryBits = COLLIDE_CATEGORY_STAGE;
//    d2.friction = 1;
//    d2.restitution = 0;
//    d2.density = 1;
//    d2.shape = &circle2;
//
    printf("Wanderer: create fixture defs \n");
    b2CircleShape circle1;
    circle1.m_radius = _SETTINGS->stage.goal_size / 2.f;
    circle1.m_p.Set(0.05,0.1);
    b2FixtureDef d1;
    d1.filter.categoryBits = COLLIDE_CATEGORY_STAGE;
    d1.friction = 1;
    d1.restitution = 0;
    d1.density = 1;
    d1.shape = &circle1;
    b2CircleShape circle2;
    circle2.m_radius = _SETTINGS->stage.goal_size / 2.f;
    circle2.m_p.Set(-0.05,-0.1);
    b2FixtureDef d2;
    d2.filter.categoryBits = COLLIDE_CATEGORY_STAGE;
    d2.friction = 1;
    d2.restitution = 0;
    d2.density = 1;
    d2.shape = &circle2;
    printf("Wanderer: create fixtures \n");
    _body1->CreateFixture(&d1);
    _body1->CreateFixture(&d2);
    printf("Wanderer: created \n");

//    b2DistanceJointDef *distance_Joint = new b2DistanceJointDef();
//    distance_Joint->bodyA = _body1;
//    distance_Joint->bodyB = _body2;
//    distance_Joint->length = 0.2;
//    distance_Joint->frequencyHz= 0.5;
//    distance_Joint->dampingRatio= 0;
//    distance_Joint->collideConnected = false;
//    w->CreateJoint(distance_Joint);

    updateVelocity();
}



void WandererBipedal::reset(const b2Vec2 & position)
{
	_body1->SetTransform(position, 0);
	_body1->SetLinearVelocity(b2Vec2_zero);
	_body1->SetAngularVelocity(0);
//    _body2->SetTransform(position, 0);
//    _body2->SetLinearVelocity(b2Vec2_zero);
//    _body2->SetAngularVelocity(0);
	updateVelocity();
}

WandererBipedal::~WandererBipedal()
{
	_body1->GetWorld()->DestroyBody(_body1);
	_body1 = NULL;
//    _body2->GetWorld()->DestroyBody(_body2);
//    _body2 = NULL;
}

void WandererBipedal::update()
{
	if(f_random() < _changeRate)
		updateVelocity();
}

void WandererBipedal::updateVelocity()
{
    float max_angle = f_rad(30);
    float max_velocity = _velocity;
    b2Vec2 v = _body1->GetLinearVelocity();
    float angle;
    zVector2D v_rot;


    b2Fixture *fixOld1 = _body1->GetFixtureList();
    _body1->DestroyFixture(fixOld1);
    fixOld1 = _body1->GetFixtureList();
    _body1->DestroyFixture(fixOld1);


    b2CircleShape circle1;
    float y_val = fmod( _counter,  2*M_PI);
    //printf("y_val : %f", sin(y_val));
    circle1.m_radius = _SETTINGS->stage.goal_size / 2.f;
    circle1.m_p.Set(sin(y_val)*0.1, 0.05);
    b2FixtureDef d1;
    d1.filter.categoryBits = COLLIDE_CATEGORY_STAGE;
    d1.friction = 1;
    d1.restitution = 0;
    d1.density = 1;
    d1.shape = &circle1;
    b2CircleShape circle2;
    circle2.m_radius = _SETTINGS->stage.goal_size / 2.f;
    circle2.m_p.Set(sin(y_val+ M_PI)*0.1, -0.05);
    b2FixtureDef d2;
    d2.filter.categoryBits = COLLIDE_CATEGORY_STAGE;
    d2.friction = 1;
    d2.restitution = 0;
    d2.density = 1;
    d2.shape = &circle2;
    _body1->CreateFixture(&d1);
    _body1->CreateFixture(&d2);
    if(_counter > 100*M_PI){
        _counter = 0;
    }else{
        _counter = _counter + 0.3;
    }

    //if (_counter %7 < 4){
//        b2CircleShape circle1;
//        circle1.m_radius = _SETTINGS->stage.goal_size / 2.f;
//        circle1.m_p.Set(0.05,0.1);
//        b2FixtureDef d1;
//        d1.filter.categoryBits = COLLIDE_CATEGORY_STAGE;
//        d1.friction = 1;
//        d1.restitution = 0;
//        d1.density = 1;
//        d1.shape = &circle1;
//        b2CircleShape circle2;
//        circle2.m_radius = _SETTINGS->stage.goal_size / 2.f;
//        circle2.m_p.Set(-0.05,-0.1);
//        b2FixtureDef d2;
//        d2.filter.categoryBits = COLLIDE_CATEGORY_STAGE;
//        d2.friction = 1;
//        d2.restitution = 0;
//        d2.density = 1;
//        d2.shape = &circle2;
//        _body1->CreateFixture(&d1);
//        _body1->CreateFixture(&d2);
//    } else {
//        b2CircleShape circle3;
//        circle3.m_radius = _SETTINGS->stage.goal_size / 2.f;
//        circle3.m_p.Set(-0.05,0.1);
//        b2FixtureDef d3;
//        d3.filter.categoryBits = COLLIDE_CATEGORY_STAGE;
//        d3.friction = 1;
//        d3.restitution = 0;
//        d3.density = 1;
//        d3.shape = &circle3;
//        b2CircleShape circle4;
//        circle4.m_radius = _SETTINGS->stage.goal_size / 2.f;
//        circle4.m_p.Set(0.05,-0.1);
//        b2FixtureDef d4;
//        d4.filter.categoryBits = COLLIDE_CATEGORY_STAGE;
//        d4.friction = 1;
//        d4.restitution = 0;
//        d4.density = 1;
//        d4.shape = &circle4;
//        _body1->CreateFixture(&d3);
//        _body1->CreateFixture(&d4);
//    }
//    _counter++;

    //printf("calc velocity\n");
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
    //printf("set velocity\n");
    _body1->SetLinearVelocity(b2Vec2(v_rot.x, v_rot.y));
    _body1->SetTransform(_body1->GetPosition(), atan2(v_rot.y,v_rot.x));
    //printf("update DONE\n");
}
