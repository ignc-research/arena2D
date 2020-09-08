#include "Wanderer.hpp"

Wanderer::Wanderer(b2World * w, const b2Vec2 & position,
					float velocity, float change_rate, float stop_rate, float max_angle_velo, unsigned int type)
{
	_velocity = velocity;
	_changeRate = change_rate;
	_stopRate = stop_rate;
	_type = type;
	_maxAngleVel = max_angle_velo;

	// creating body
	b2BodyDef body_def;
	body_def.type = b2_dynamicBody;
	body_def.allowSleep = false;
	body_def.position = position;
	body_def.linearDamping = 0;
	body_def.angularDamping = 0.1;
	body_def.userData = (void*)this;
	_body = w->CreateBody(&body_def);

	// initial velocity update
//	updateVelocity();
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

void Wanderer::reset(const b2Vec2 & position)
{
	_body->SetTransform(position, 0);
	_body->SetLinearVelocity(b2Vec2_zero);
	_body->SetAngularVelocity(0);
	updateVelocity();
}

Wanderer::~Wanderer()
{
	_body->GetWorld()->DestroyBody(_body);
	_body = NULL;
}

void Wanderer::update(bool chat_flag)
{
	if(f_random() <= _changeRate)
		updateVelocity();
}

void Wanderer::updateVelocity()
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
