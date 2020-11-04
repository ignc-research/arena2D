#include "Wanderer.hpp"
#include "Wanderers.hpp"
#include <iostream>
Wanderer::Wanderer(b2World * w, const b2Vec2 & position, float velocity, unsigned int type, unsigned int mode,
			std::vector<b2Vec2> waypoints, int stop_counter_threshold, float change_rate, float stop_rate, float max_angle_velo)
{	
	// general param
	_initPosition = position;
	_velocity = velocity;
	_type = type;
	_mode= mode;

	// path follow param
	_waypoints=waypoints;
	_stopCounterThreshold=stop_counter_threshold;
	
	_stopCounter=_stopCounterThreshold;
	_timeOutCounter=0;
	_indexWaypoint=0;
	_directForward=true;

	// random location param
	_changeRate = change_rate;
	_stopRate = stop_rate;
	_type = type;

	_maxAngleVel = max_angle_velo;

	// creating body
	b2BodyDef body_def;
	if (type == WANDERER_ID_HUMAN){
        body_def.type = b2_dynamicBody_human;
	} else {
        body_def.type = b2_dynamicBody;
    }
	body_def.allowSleep = false;
	body_def.position = position;
	body_def.linearDamping = 0;
	body_def.angularDamping = 0.1;
	body_def.userData = (void*)this;
	_body = w->CreateBody(&body_def);

}

void Wanderer::addCircle(float radius, const b2Vec2 & pos)
{
	b2CircleShape circle;
	circle.m_radius = radius;
	circle.m_p = pos;
	b2FixtureDef d;
	d.filter.categoryBits = COLLIDE_CATEGORY_STAGE;
	d.friction = 1;
	d.restitution = 0;
	d.density = 1;
	d.shape = &circle;
	_body->CreateFixture(&d);
}

void Wanderer::addRobotPepper(float length_triangle)
{
	// base
	float sqr3=sqrt(3);
	float r=0.5/(1+sqr3+1)*length_triangle;
	b2PolygonShape triangle_shape;
	b2Vec2 verts_triangle[6];
	
	/*
	verts_triangle[0].Set(0.0f, length_triangle*sqrt(3)/2); 
	verts_triangle[1].Set(length_triangle/2, 0); 
	verts_triangle[2].Set(-length_triangle/2, 0); 
	triangle_shape.Set(verts_triangle, 3);
	*/
	
	
	verts_triangle[0].Set(0.5*length_triangle-sqr3*r,			0.0f); 
	verts_triangle[1].Set(-0.5*length_triangle+sqr3*r,			0.0f); 
	verts_triangle[2].Set(-0.5*length_triangle+0.5*sqr3*r,  	0.866*r); 
	verts_triangle[3].Set(-0.5*sqr3*r,						0.5*sqr3*length_triangle-2*r); 
	verts_triangle[4].Set( 0.5*sqr3*r,						0.5*sqr3*length_triangle-2*r); 
	verts_triangle[5].Set(0.5*length_triangle-0.5*sqr3*r,	0.5*sqr3*r); 
	triangle_shape.Set(verts_triangle,6);
	


	b2FixtureDef fix;
	fix.filter.categoryBits = COLLIDE_CATEGORY_PLAYER;
	fix.shape = &triangle_shape;
	fix.density = 1;
	fix.friction = 1;
	fix.restitution = 0;

	_body->CreateFixture(&fix);

	// circle
	for(int i=0;i<3;i++)
	{
		b2CircleShape circle;
		circle.m_radius = r;
		b2Vec2 center;
		switch(i) {
  			case 0:
    			center.Set(0.5*length_triangle-1.732*circle.m_radius,circle.m_radius);
    			break;
  			case 1:
    			center.Set(-0.5*length_triangle+1.732*circle.m_radius,circle.m_radius);
    			break;
			case 2:
    			center.Set(0.0,0.866*length_triangle-1.732*circle.m_radius);
    			break;
  			
		}
		circle.m_p=center;

		b2FixtureDef d;
		d.filter.categoryBits = COLLIDE_CATEGORY_PLAYER;
		d.friction = 1;
		d.restitution = 0;
		d.density = 1;
		d.shape = &circle;
		_body->CreateFixture(&d);

	}
}

void Wanderer::reset(const b2Vec2 & position)
{
	_body->SetTransform(position, 0);
	_body->SetLinearVelocity(b2Vec2_zero);
	_body->SetAngularVelocity(0);

	_stopCounter=_stopCounterThreshold;
	_timeOutCounter=0;
	_indexWaypoint=0;
	_directForward=true;


	if(_mode==0)// random mode
	{
		updateVelocityRandomMode();

	}
	else if(_mode==1) // path follow mode
	{
		updateVelocityPathMode();
	}
}

Wanderer::~Wanderer()
{
	_body->GetWorld()->DestroyBody(_body);
	_body = NULL;
}

void Wanderer::update(bool chat_flag)
{	
	if(_mode==0) // random mode
	{
		if(f_random() <= _changeRate)
		{
			updateVelocityRandomMode();
		}
	}
	else if(_mode==1)// path follow mode
	{
		updateVelocityPathMode();
	}
	
}

void Wanderer::updateVelocityRandomMode()
{
	float max_angle = f_rad(_maxAngleVel);
	float max_velocity = _velocity;
	b2Vec2 v = _body->GetLinearVelocity();
	float angle = _body->GetAngle();
	float angle_velocity = 0;
	if(f_random() < _stopRate){// stop wanderer
		v.Set(0,0);
	}
	else{
		float sign = 1;
		if(fabs(v.x) < 0.001 && fabs(v.y) < 0.001 && rand()%2 == 0)
			sign = -1;
		zVector2D _v = zVector2D(0, sign*max_velocity).getRotated(angle);
		v.Set(_v.x, _v.y);
		angle_velocity += f_frandomRange(-max_angle, max_angle);
	}
	_body->SetAngularVelocity(angle_velocity);
	_body->SetLinearVelocity(v);
}



void Wanderer::updateVelocityPathMode(){

	b2Vec2 v = _body->GetLinearVelocity();
	b2Vec2 currentPosition = _body->GetPosition();
	b2Vec2 nextPosition =_waypoints[_indexWaypoint];

	float distanceToNext=b2Distance(currentPosition, nextPosition);

	bool reached=distanceToNext<NEAR_REGION_DISTANCE;

	//std::cout<<"Distance:"<<distanceToNext<<std::endl;

	if(reached)
	{// reached
		if(_stopCounter> _stopCounterThreshold){
			// arrived and  stay for a while
			_stopCounter++;
			//std::cout<<"Reach & stop:"<<_stopCounter<<std::endl;
		}
		else
		{ 	// select next waypoint to head to
			if(_indexWaypoint>=_waypoints.size()-1) // if current index is the last
			{
				_directForward=false;
			}
			else if(_indexWaypoint<=0) // if current index is the first
			{
				_directForward=true;
			}

			if(_directForward){
				_indexWaypoint++;
			}
			else
			{
				_indexWaypoint--;
			}
			//std::cout<<"Reach & next waypoint:"<<_indexWaypoint<<std::endl;

			_stopCounter=0;
		}
		v.Set(0,0);
		_timeOutCounter=0;

	}
	else
	{//not reached yet

		if(_timeOutCounter>20){
			// select next waypoint to head to
			if(_indexWaypoint>=_waypoints.size()-1) // if current index is the last
			{
				_directForward=false;
			}
			else if(_indexWaypoint<=0) // if current index is the first
			{
				_directForward=true;
			}

			if(_directForward){
				_indexWaypoint++;
			}
			else
			{
				_indexWaypoint--;
			}
			_timeOutCounter=0;
			
			

		}
		else
		{
			// calculate the velocity direction
			zVector2D v_ = zVector2D(nextPosition.x-currentPosition.x, nextPosition.y-currentPosition.y);
			float a = v_.getRotation();
			zVector2D v_set = zVector2D(_velocity,0 ).getRotated(a);
			// set velocity
			v.Set(v_set.x,v_set.y);
			if(abs(_distanceToNext_pre-distanceToNext)<0.01){
				_timeOutCounter++;
			}
			
			

			//std::cout<<"go forward to waypoint"<<_indexWaypoint<<std::endl;
			//std::cout<<"set v x"<<v.x<<" y "<<v.y<<std::endl;
			//std::cout<<"time out counter"<<_timeOutCounter<<std::endl;
			_distanceToNext_pre=distanceToNext;
		}
	}

	_body->SetAngularVelocity(0.0f);
	_body->SetLinearVelocity(v);

};
