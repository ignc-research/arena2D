#include "PhysicsWorld.hpp"

const b2Vec2 PHYSICS_NEIGHBOUR_MAP[4] = {b2Vec2(1, 0), b2Vec2(0,1), b2Vec2(-1, 0), b2Vec2(0, -1)};
#define SQRT_2_OVER_2 0.7071067812
const b2Vec2 PHYSICS_NEIGHBOUR_MAP_DIAGONAL[4] = {b2Vec2(SQRT_2_OVER_2, SQRT_2_OVER_2), b2Vec2(-SQRT_2_OVER_2,SQRT_2_OVER_2), b2Vec2(-SQRT_2_OVER_2, -SQRT_2_OVER_2), b2Vec2(SQRT_2_OVER_2, -SQRT_2_OVER_2)};

PhysicsWorld::PhysicsWorld()
{
	_world = NULL;
	_velocityIterations = 8;
	_positionIterations = 3;

	//setting default colors
	_fillAlpha = 1.0f;
	_dynamicColor.set(0x4ea5ffFF);
	_dynamicColor.brighten(0.3);
	_sensorColor.set(0xf3c355FF);
	_sleepColor.set(0x999999FF);
	_staticColor.set(0x000000FF);
	_staticColor2.set(0xBBBBBBFF);
	_kinematicColor.set(0x222222FF);
	_collisionsColor.set(0x21f978FF);
	_staticBody = NULL;
}

PhysicsWorld::~PhysicsWorld()
{
	delete _world;
}

void PhysicsWorld::init()
{
	_world = new b2World(b2Vec2(0.f, 0.f));
	//creating single static body
	b2BodyDef d;
	d.type = b2_staticBody;
	d.position.Set(0,0);
	_staticBody = _world->CreateBody(&d);
}

void PhysicsWorld::step(float time_step)
{
	_world->Step(time_step, _velocityIterations, _positionIterations);
}

void PhysicsWorld::calculateVisibleFixturesWorld(b2World * w, const zRect & area)
{
	//getting all visible objects through aabb query
	_aabbQuery.reset();
	b2AABB aabb;
	aabb.lowerBound = b2Vec2(area.x-area.w, area.y-area.h);
	aabb.upperBound = b2Vec2(area.x+area.w, area.y+area.h);
	w->QueryAABB((b2QueryCallback*)&_aabbQuery, aabb);
}

void PhysicsWorld::debugDrawWorld(b2World * w, unsigned int flags, uint16 category_bits)
{
	Colorplex2DShader * colorplex = _RENDERER->getColorplexShader();
	bool isColorplex = colorplex == Z_SHADER;
	if(isColorplex)
	{
		colorplex->setColor2(_staticColor2);
		colorplex->setPatternSize(3);
	}
	Colorplex2DShader::PatternType colorplex_type = Colorplex2DShader::DIAGONAL_POSITIVE;

	//going through all visible fixtures
	std::list<b2Fixture*>::iterator it = _aabbQuery._visibleFixtures.begin();
	std::list<b2Fixture*>::iterator it_end = _aabbQuery._visibleFixtures.end();
	zMatrix4x4 modelview;
	glLineWidth(1.f);
	for(;it != it_end; it++)
	{
		b2Fixture * f= *it;
		b2Filter filter = f->GetFilterData();
		if((filter.categoryBits & category_bits) == 0
			|| (filter.categoryBits & COLLIDE_CATEGORY_DONT_RENDER)){// not in category bits or not render category
			continue;
		}
		b2Body * b =f->GetBody();
		b2Shape * s = f->GetShape();
		zColor color;
		bool setColorplex = false;
		//setting color
		if(f->IsSensor())
		{
			if((flags & PHYSICS_RENDER_SENSORS) == 0)
				continue;
			color = _sensorColor;
		}
		else
		{
			switch(b->GetType())
			{
			case b2_dynamicBody:
			{
				if((flags & PHYSICS_RENDER_DYNAMIC) == 0)
					continue;
				if(b->IsAwake())
					color = _dynamicColor;
				else
					color = _sleepColor;
			}break;
			case b2_staticBody:
			{
				if((flags & PHYSICS_RENDER_STATIC) == 0)
					continue;
				setColorplex = isColorplex;
				color = _staticColor;
			}break;
			case b2_kinematicBody:
			{
				color = _kinematicColor;
			}break;
			}
		}
		zColor dark = color;
		dark.darken(0.4);
		b2Transform t = b->GetTransform();
		modelview.setTranslation(zVector3D(t.p.x, t.p.y, 0.f));
		zMatrix4x4 rotation;
		rotation.setRotationZ(t.q.GetAngle()*180/M_PI);
		modelview = modelview*rotation;
		switch(s->GetType())
		{
		case b2Shape::e_polygon:
		{
			b2PolygonShape * _s = (b2PolygonShape*)s;
			if(_s->m_count == 4 && _s->m_vertices[3].x == _s->m_vertices[0].x && _s->m_vertices[0].y == _s->m_vertices[1].y &&
									_s->m_vertices[1].x == _s->m_vertices[2].x && _s->m_vertices[2].y == _s->m_vertices[3].y)//check whether shape is a box for faster rendering
			{
				float dimX = _s->m_vertices[1].x - _s->m_vertices[3].x;
				float dimY = _s->m_vertices[3].y - _s->m_vertices[0].y;
				//render quad
				zMatrix4x4 scale;
				zMatrix4x4 translate;
				scale.setScale(zVector3D(dimX, dimY, 0.f));
				translate.setTranslation(zVector3D(_s->m_centroid.x, _s->m_centroid.y, 0.f));
				modelview = modelview*translate*scale;
				Z_SHADER->setModelviewMatrix(modelview);
				color.a = _fillAlpha;
				if(setColorplex)
					colorplex->setPatternType(colorplex_type);
				Z_SHADER->setColor((float*)&color);
				_RENDERER->bindGO(GO_CENTERED_QUAD);
				_RENDERER->drawGO();
				if(setColorplex)
					colorplex->setPatternType(Colorplex2DShader::NONE);
				Z_SHADER->setColor((float*)&dark);
				_RENDERER->bindGO(GO_CENTERED_QUAD_LINE);
				_RENDERER->drawGO();
			}
			else//shape is complex polygon
			{
				GLuint buffer;
				glGenBuffers(1, &buffer);
				glBindBuffer(GL_ARRAY_BUFFER, buffer);
				glBufferData(GL_ARRAY_BUFFER, sizeof(float)*2*_s->m_count, _s->m_vertices, GL_STATIC_DRAW);
				Z_SHADER->enableVertexArray(true);
				Z_SHADER->setVertexAttribPointer2D(0,0);
				Z_SHADER->setModelviewMatrix((float*)&modelview);
				color.a = _fillAlpha;
				Z_SHADER->setColor((float*)&color);
				if(setColorplex)
					colorplex->setPatternType(colorplex_type);
				glDrawArrays(GL_POLYGON, 0, _s->m_count);
				if(setColorplex)
					colorplex->setPatternType(Colorplex2DShader::NONE);
				Z_SHADER->setColor((float*)&dark);
				glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
				glDrawArrays(GL_POLYGON, 0, _s->m_count);
				glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
				Z_SHADER->enableVertexArray(false);
				glBindBuffer(GL_ARRAY_BUFFER, 0);
				glDeleteBuffers(1, &buffer);
			}
		}break;
		case b2Shape::e_circle:
		{
			b2CircleShape * c = (b2CircleShape*)s;
			zMatrix4x4 scale;
			scale.setScale(zVector3D(c->m_radius, c->m_radius, 0.f));
			zMatrix4x4 translate;
			translate.setTranslation(zVector3D(c->m_p.x, c->m_p.y, 0.f));
			modelview = modelview*translate*scale;
			Z_SHADER->setModelviewMatrix(modelview);
			Z_SHADER->setColor((float*)&color);
			_RENDERER->bindGO(GO_CIRCLE);
			if(setColorplex)
				colorplex->setPatternType(colorplex_type);
			_RENDERER->drawGO();
			if(setColorplex)
				colorplex->setPatternType(Colorplex2DShader::NONE);
			Z_SHADER->setColor((float*)&dark);
			_RENDERER->bindGO(GO_CIRCLE_LINE);
			_RENDERER->drawGO();
		}break;
		case b2Shape::e_edge:
		{
			b2EdgeShape * e = (b2EdgeShape*)s;
			zMatrix4x4 scale;
			zMatrix4x4 translate;
			translate.setTranslation(zVector3D(e->m_vertex1.x, e->m_vertex1.y, 0.f));
			b2Vec2 dif = e->m_vertex2 - e->m_vertex1;
			scale.setScale(zVector3D(dif.x, dif.y, 0.f));
			modelview = modelview*translate*scale;
			Z_SHADER->setModelviewMatrix((float*)&modelview);
			Z_SHADER->setColor((float*)&color);
			_RENDERER->bindGO(GO_LINE);
			_RENDERER->drawGO();
		}break;
		case b2Shape::e_chain:
		{
			b2ChainShape * _s = (b2ChainShape*)s;
			if(_s->m_count == 5 && _s->m_vertices[3].x == _s->m_vertices[0].x && _s->m_vertices[0].y == _s->m_vertices[1].y &&
									_s->m_vertices[1].x == _s->m_vertices[2].x && _s->m_vertices[2].y == _s->m_vertices[3].y)//check whether shape is a box for faster rendering
			{
				float dimX = _s->m_vertices[1].x - _s->m_vertices[3].x;
				float dimY = _s->m_vertices[3].y - _s->m_vertices[0].y;
				//render quad
				zMatrix4x4 scale;
				scale.setScale(zVector3D(dimX, dimY, 0.f));
				modelview = modelview*scale;
				Z_SHADER->setModelviewMatrix(modelview);
				Z_SHADER->setColor((float*)&color);
				_RENDERER->bindGO(GO_CENTERED_QUAD_LINE);
				_RENDERER->drawGO();
			}
			else//shape is complex polygon
			{
				GLuint buffer;
				glGenBuffers(1, &buffer);
				glBindBuffer(GL_ARRAY_BUFFER, buffer);
				glBufferData(GL_ARRAY_BUFFER, sizeof(float)*2*_s->m_count, _s->m_vertices, GL_STATIC_DRAW);
				Z_SHADER->enableVertexArray(true);
				Z_SHADER->setVertexAttribPointer2D(0,0);
				Z_SHADER->setModelviewMatrix(modelview);
				Z_SHADER->setColor((float*)&color);
				glDrawArrays(GL_LINE_STRIP, 0, _s->m_count);
				Z_SHADER->enableVertexArray(false);
				glBindBuffer(GL_ARRAY_BUFFER, 0);
				glDeleteBuffers(1, &buffer);
			}
		}break;
		default:break;
		}
	}

	glLineWidth(1.f);
	//resetting pattern
	if(isColorplex)
		colorplex->setPatternType(Colorplex2DShader::NONE);

	//resetting modelview
	_RENDERER->resetModelviewMatrix();
}
