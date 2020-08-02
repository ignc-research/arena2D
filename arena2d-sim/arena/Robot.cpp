/* Author: Cornelius Marx */
#include "Robot.hpp"


Robot::Robot(b2World * world): _lidarBuffer(0){
	b2BodyDef b;
	b.allowSleep = false;
	b.fixedRotation = false;
	b.type = b2_dynamicBody;
	b2FixtureDef fix;
	fix.filter.categoryBits = COLLIDE_CATEGORY_PLAYER;

	// main body
	b.position = b2Vec2(0, 0);
	_base = world->CreateBody(&b);
	b2PolygonShape shape;
	b2Vec2 verts[8];
	float distx = _SETTINGS->robot.base_size.x/2.f;
	float disty = _SETTINGS->robot.base_size.y/2.f;
	float distx2 = distx-_SETTINGS->robot.bevel_size.x;
	float disty2 = disty-_SETTINGS->robot.bevel_size.y;
	verts[0].Set(-distx2, disty); 
	verts[1].Set(-distx, disty2);
	verts[2].Set(-distx, -disty2);
	verts[3].Set(-distx2, -disty);
	verts[4].Set(distx2, -disty);
	verts[5].Set(distx, -disty2);
	verts[6].Set(distx, disty2);
	verts[7].Set(distx2, disty);
	shape.Set(verts, 8);
	fix.shape = &shape;
	fix.density = 1;
	fix.friction = 1;
	fix.restitution = 0;

	_base->CreateFixture(&fix);


	// calculate wheel positions
	b2PolygonShape box_shape;
	fix.shape = &box_shape;
	fix.density = 1;
	float wheel_w = _SETTINGS->robot.wheel_size.x;
	float wheel_h = _SETTINGS->robot.wheel_size.y;
	float wheel_half_w = wheel_w/2.f;
	float wheel_half_h = wheel_h/2.f;
	float wheel_x = _SETTINGS->robot.wheel_offset.x;
	float wheel_y = _SETTINGS->robot.wheel_offset.y;
	_wheelDistance = distx + wheel_half_w + wheel_x;
	_wheelPosition[RIGHT].Set(_wheelDistance, wheel_y);
	_wheelPosition[LEFT].Set(-_wheelDistance, wheel_y);

	_wheelDistance *= 2.f;
	// right wheel
	box_shape.SetAsBox(wheel_half_w, wheel_half_h, _wheelPosition[RIGHT], 0);
	_base->CreateFixture(&fix);

	// left wheel
	box_shape.SetAsBox(wheel_half_w, wheel_half_h, _wheelPosition[LEFT], 0);
	_base->CreateFixture(&fix);

	// calculate safe radius
	_safeRadius = verts[0].Length();
	_safeRadius = f_fmax(_safeRadius, verts[1].Length());
	_safeRadius = f_fmax(_safeRadius, b2Vec2(distx+wheel_w+wheel_x, wheel_y+wheel_half_h).Length());
	_safeRadius = f_fmax(_safeRadius, b2Vec2(distx+wheel_w+wheel_x, wheel_y-wheel_half_h).Length());
	_safeRadius += 0.01f;

	// safety radius sensor
	b2CircleShape circle_shape;
	circle_shape.m_radius = _safeRadius;
	fix.shape = &circle_shape;
	fix.filter.categoryBits |= COLLIDE_CATEGORY_DONT_RENDER | COLLIDE_CATEGORY_PLAYER;
	fix.filter.maskBits = COLLIDE_CATEGORY_STAGE | COLLIDE_CATEGORY_GOAL;
	fix.isSensor = true;
	_safetyRadiusSensor = _base->CreateFixture(&fix);

	_contactCount = 0;

	_lidar = NULL;
	updateLidar();

	// generating buffer for laser data and trail
	if(_SETTINGS->video.enabled){
		// laser data
		glGenBuffers(1, &_lidarBuffer);
		_lidarBufferCount = 0;
		
		// trail
		glGenBuffers(1, &_trailBuffer);
		glBindBuffer(GL_ARRAY_BUFFER, _trailBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float)*2*BURGER_TRAIL_BUFFER_SIZE, NULL, GL_DYNAMIC_DRAW);
		resetTrail();
	}
}

void Robot::updateLidar()
{
	// creating lidar scanner
	delete _lidar;
	const f_robotSettings & s = _SETTINGS->robot;
	_lidar = new LidarCast(	s.laser_num_samples, s.laser_max_distance,
							f_rad(s.laser_start_angle), f_rad(s.laser_end_angle),
							s.laser_noise, _base);
}

Robot::~Robot()
{
	if(_SETTINGS->video.enabled){
		glDeleteBuffers(1, &_lidarBuffer);
		glDeleteBuffers(1, &_trailBuffer);
	}

	// destroy lidar
	delete _lidar;

	// destroy body
	_base->GetWorld()->DestroyBody(_base);
}

void Robot::reset(const b2Vec2 & position, float angle)
{
	_base->SetLinearVelocity(b2Vec2_zero);
	_base->SetAngularVelocity(0);
	_base->SetTransform(position, angle);
	_contactCount = 0;

	if(_SETTINGS->video.enabled)
		resetTrail();
}

void Robot::resetTrail()
{
	_trailVertexCount = 1;
	_lastTrailPosition = _base->GetTransform().p;
	glBindBuffer(GL_ARRAY_BUFFER, _trailBuffer);
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float)*2, &_lastTrailPosition);
}

void Robot::updateTrail()
{
	if(_trailVertexCount >= BURGER_TRAIL_BUFFER_SIZE)// no more space for vertices
		return;

	b2Vec2 pos = _base->GetTransform().p;
	if((pos-_lastTrailPosition).Length() > BURGER_TRAIL_SEGMENT_SIZE)// enough distance from last vertex
	{
		glBindBuffer(GL_ARRAY_BUFFER, _trailBuffer);
		glBufferSubData(GL_ARRAY_BUFFER, _trailVertexCount*sizeof(float)*2, sizeof(float)*2, &pos);
		_lastTrailPosition = pos;
		_trailVertexCount++;
	}
}

void Robot::getActionTwist(Action a, Twist & t)
{
	const f_robotSettings & s = _SETTINGS->robot;
	switch(a)	
	{
	case FORWARD:
	{
		t.linear = s.forward_speed.linear;
		t.angular = s.forward_speed.angular;
	}break;
	case FORWARD_LEFT:
	{
		t.linear = s.left_speed.linear;
		t.angular = s.left_speed.angular;
	}break;
	case FORWARD_RIGHT:
	{
		t.linear = s.right_speed.linear;
		t.angular = s.right_speed.angular;
	}break;
	case FORWARD_STRONG_LEFT:
	{
		t.linear = s.strong_left_speed.linear;
		t.angular = s.strong_left_speed.angular;
	}break;
	case FORWARD_STRONG_RIGHT:
	{
		
		t.linear = s.strong_right_speed.linear;
		t.angular = s.strong_right_speed.angular;
	}break;
	case BACKWARD:
	{
		t.linear = s.backward_speed.linear;
		t.angular = s.backward_speed.angular;
	}break;
	case STOP:
	default:
	{
		t.linear = 0;
		t.angular = 0;
	}break;
	}
}

void Robot::performAction(Action a)
{
	Twist t;
	getActionTwist(a, t);
	performAction(t);
}


void Robot::scan()
{
	b2Transform t = _base->GetTransform();
	b2Vec2 center = _base->GetWorldPoint(b2Vec2(_SETTINGS->robot.laser_offset.x, _SETTINGS->robot.laser_offset.y)); 
	_lidar->scan(_base->GetWorld(), center, t.q.GetAngle());
}

void Robot::renderScan(bool area)
{
	_lidarBufferCount = _lidar->getNumSamples()+2;
	const b2Vec2 * samples = _lidar->getPointsWithCenter();
	glBindBuffer(GL_ARRAY_BUFFER, _lidarBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(b2Vec2)*_lidarBufferCount, samples, GL_STATIC_DRAW);

	_RENDERER->resetModelviewMatrix();
	glPointSize(BURGER_LIDAR_POINT_SIZE);
	glBindBuffer(GL_ARRAY_BUFFER, _lidarBuffer);
	GLint vertex_loc = Z_SHADER->getVertexLoc();
	glEnableVertexAttribArray(vertex_loc);
	glVertexAttribPointer(vertex_loc, 2, GL_FLOAT, GL_FALSE, 0, 0);
	Z_SHADER->setColor(zColor(BURGER_LIDAR_POINT_COLOR));
	glDrawArrays(GL_POINTS, 1, _lidarBufferCount-2);
	if(area){
		Z_SHADER->setColor(zColor(BURGER_LIDAR_AREA_COLOR));
		glDrawArrays(GL_TRIANGLE_FAN, 0, _lidarBufferCount);
	}

	// restore
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDisableVertexAttribArray(vertex_loc);
	glPointSize(1);
}


void Robot::renderTrail()
{
	_RENDERER->resetModelviewMatrix();
	glLineWidth(BURGER_TRAIL_WIDTH);
	glBindBuffer(GL_ARRAY_BUFFER, _trailBuffer);
	GLint vertex_loc = Z_SHADER->getVertexLoc();
	glEnableVertexAttribArray(vertex_loc);
	glVertexAttribPointer(vertex_loc, 2, GL_FLOAT, GL_FALSE, 0, 0);
	Z_SHADER->setColor(zColor(BURGER_TRAIL_COLOR));
	glDrawArrays(GL_LINE_STRIP, 0, _trailVertexCount);

	// restore
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDisableVertexAttribArray(vertex_loc);
	glLineWidth(1);
}

void Robot::performAction(const Twist & t)
{
	float left_speed = t.linear - t.angular*_wheelDistance/2.0f;
	float right_speed = t.linear + t.angular*_wheelDistance/2.0f;
	float rad = _base->GetAngle();
	float sin_rad = sin(rad);
	float cos_rad = cos(rad);
	b2Vec2 dir(-sin_rad, cos_rad);
	b2Vec2 codir(cos_rad, sin_rad);
	float mass = _base->GetMass();
	
	b2Vec2 current_vel_left = _base->GetLinearVelocityFromLocalPoint(_wheelPosition[LEFT]);
	b2Vec2 current_vel_right = _base->GetLinearVelocityFromLocalPoint(_wheelPosition[RIGHT]);

	b2Vec2 vel_left = left_speed*dir - current_vel_left;
	b2Vec2 vel_right = right_speed*dir - current_vel_right;

	// apply force to left wheel
	b2Vec2 force_left = 5*mass*vel_left;
	_base->ApplyForce(force_left, _base->GetWorldPoint(_wheelPosition[LEFT]), false);

	// apply force to right wheel
	b2Vec2 force_right = 5*mass*vel_right;
	_base->ApplyForce(force_right, _base->GetWorldPoint(_wheelPosition[RIGHT]), false);
}
