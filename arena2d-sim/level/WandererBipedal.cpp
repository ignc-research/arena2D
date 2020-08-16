#include "WandererBipedal.hpp"

WandererBipedal::WandererBipedal(b2World * w, float radius, const b2Vec2 & position,
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

	b2CircleShape circle1;
	circle1.m_radius = radius;
	circle1.m_p.Set(-radius-0.025,0);
	b2FixtureDef d1;
	d1.filter.categoryBits = COLLIDE_CATEGORY_STAGE;
	d1.friction = 1;
	d1.restitution = 0;
	d1.density = 1;
	d1.shape = &circle1;


    b2CircleShape circle2;
    circle2.m_radius = radius;
    circle2.m_p.Set(radius+0.025,0);
    b2FixtureDef d2;
    d2.filter.categoryBits = COLLIDE_CATEGORY_STAGE;
    d2.friction = 1;
    d2.restitution = 0;
    d2.density = 1;
    d2.shape = &circle2;

    _body->CreateFixture(&d1);
    _body->CreateFixture(&d2);
	updateVelocity();
}

void WandererBipedal::reset(const b2Vec2 & position)
{
	_body->SetTransform(position, 0);
	_body->SetLinearVelocity(b2Vec2_zero);
	_body->SetAngularVelocity(0);
	updateVelocity();
}

WandererBipedal::~WandererBipedal()
{
	_body->GetWorld()->DestroyBody(_body);
	_body = NULL;
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
