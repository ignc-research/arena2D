#include "Wanderer.hpp"

Wanderer::Wanderer(b2World * w, float radius, const b2Vec2 & position,
					float velocity, float change_rate, float stop_rate, unsigned int type)
{
	_velocity = velocity;
	_changeRate = change_rate;
	_stopRate = stop_rate;
	_type = type;
	_radius = radius;

	// creating body
	b2BodyDef body_def;
	body_def.type = b2_dynamicBody;
	body_def.allowSleep = false;
	body_def.position = position;
	body_def.linearDamping = 0;
	body_def.angularDamping = 0;
	body_def.userData = (void*)this;
	_body = w->CreateBody(&body_def);

	b2CircleShape circle;
	circle.m_radius = radius;
	circle.m_p.Set(0,0);
	b2FixtureDef d;
	d.filter.categoryBits = COLLIDE_CATEGORY_STAGE;
	d.friction = 1;
	d.restitution = 0;
	d.density = 1;
	d.shape = &circle;
	_body->CreateFixture(&d);
	updateVelocity();
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

void Wanderer::update()
{
	if(f_random() < _changeRate)
		updateVelocity();
}

void Wanderer::updateVelocity()
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
			v_rot = zVector2D(max_velocity, 0).getRotated(angle);
		}
		else{
			angle = (max_angle*2*f_random())-max_angle;
			v_rot = max_velocity * zVector2D(v.x, v.y).getNormalized().getRotated(angle);
		}
	}
	_body->SetLinearVelocity(b2Vec2(v_rot.x, v_rot.y));
}
