/* author: Cornelius Marx */
#include "ParticleEmitter.hpp"

ParticleEmitter::ParticleEmitter()
{
	_particles = NULL;
	_particleOptions = NULL;
	_emitting = false;
}

ParticleEmitter::~ParticleEmitter()
{
	freeParticles();
}

void ParticleEmitter::freeParticles()
{
	delete[] _particles;
	_particles = NULL;
	delete[] _particleOptions;
	_particleOptions = NULL;
}

void ParticleEmitter::init(const zVector2D & pos, const ParticleEmitterOptions & options, int max_particles)
{
	freeParticles();
	_options = options;
	_pos = pos;
	_maxParticles = f_imin(max_particles, _RENDERER->getParticleShader()->getMaxNumParticles());
	_particles = new ParticleTriangle[_maxParticles];
	_particleOptions = new ParticleOptions[_maxParticles];
	_numUsedParticles = 0;
	_usedParticlesOffset = 0;
	_nextAllocationIndex = 0;
	_numAlive = 0;
	_growDirection = +1;
}

void ParticleEmitter::startEmission(){
	if(!_emitting){
		_emitting = true;
		_timeLastSpawn = Z_FW->getCurrentTime();
	}
}

void ParticleEmitter::endEmission(){
	_emitting = false;
}

void ParticleEmitter::killParticles(){
	_numUsedParticles = 0;
	_usedParticlesOffset = 0;
	_nextAllocationIndex = 0;
	_numAlive = 0;
	_growDirection = +1;
}

void ParticleEmitter::render()
{
	ParticleShader * part_shader = _RENDERER->getParticleShader();
	part_shader->setTriangles(_numUsedParticles, 0, _particles+_usedParticlesOffset);
	part_shader->draw(_numUsedParticles);
}

void ParticleEmitter::update()
{
	// updating every particle
	float delta_time = Z_FW->getDeltaTime();
	for(int i = 0; i < _numUsedParticles; i++){
		ParticleOptions * o = &_particleOptions[_usedParticlesOffset+i];
		ParticleTriangle * t = &_particles[_usedParticlesOffset+i];
		if(o->alive){
			// updating time alive
			o->time_alive += delta_time;
			if(o->time_alive >= o->fading_start_time + o->fading_time) { // time's up
				o->alive = false;
				_numAlive--;
				t->scale = 0;
				// check whether block can be tightend
				if(i == 0){// first particle in block -> raise lower limit
					int j = 0;
					int current_used_particles = _numUsedParticles;
					while(!_particleOptions[_usedParticlesOffset].alive && j < current_used_particles){
						_usedParticlesOffset++;
						_numUsedParticles--;
						j++;
					}
				}else if(i == _numUsedParticles-1){// last particle in block -> lower upper limit
					int j = _numUsedParticles-1;
					while(!_particleOptions[_usedParticlesOffset+j].alive && j >= 0){
						_numUsedParticles--;
						j--;
					}
				}
			}else{// updating "physics"
				// applying damping
				o->linear_velocity -= delta_time*o->linear_damping;
				if(o->linear_velocity < 0)
					o->linear_velocity = 0;
				o->angular_velocity -= delta_time*o->angular_damping;
				if(o->angular_velocity < 0)
					o->angular_velocity = 0;

				// applying velocity to position/rotation
				zVector2D dir(1, 0);
				dir.rotate(o->angle);
				t->x += delta_time*o->linear_velocity*dir.x;
				t->y += delta_time*o->linear_velocity*dir.y;
				t->rad += delta_time*o->angular_velocity;
				float time_percentage = o->time_alive/(o->fading_start_time+o->fading_time);
				zColor c = zColor::getInterpolated(_options.start_color, _options.end_color, time_percentage);
				t->r = c.r;
				t->g = c.g;
				t->b = c.b;
				t->a = c.a;
				float fading_time = o->time_alive-o->fading_start_time;
				if(fading_time > 0.0f){// multiplying alpha if fading time has started
					t->a *= 1-(fading_time/o->fading_time);
				}
			}
		}
	}

	// spawning new particles
	if(_emitting){
		float current_time = Z_FW->getCurrentTime();
		if(current_time - _timeLastSpawn > _options.spawn_time.min){
			spawnParticles((int)floor((current_time-_timeLastSpawn)/_options.spawn_time.min));
			_timeLastSpawn = current_time;
		}
	}
}

void ParticleEmitter::spawnParticles(int count){
	assert(_growDirection != 0);
	if(_usedParticlesOffset == 0){
		_growDirection = +1;
	}
	else if(_numUsedParticles == _maxParticles){
		_growDirection = -1;
	}
	for(int i = 0; i < count; i++){
		if(_usedParticlesOffset == 0 && _numUsedParticles == _maxParticles){ // no more space available
			return;
		}
		int new_particle_index = -1;
		if(_growDirection > 0){
			new_particle_index = _numUsedParticles+_usedParticlesOffset;
			_numUsedParticles++;
			if(new_particle_index == _maxParticles-1){// reverse grow direction
				_growDirection = -1;
			}
		}
		else if(_growDirection < 0){
			_numUsedParticles++;
			_usedParticlesOffset--;
			new_particle_index = _usedParticlesOffset;
			if(new_particle_index == 0){// reverse grow direction
				_growDirection = +1;
			}
		}

		assert(new_particle_index >= 0);
		_particles[new_particle_index].set(_pos, _options.scale.random(), 0, _options.start_color);
		_particleOptions[new_particle_index].init(_options.angle.random(), _options.linear_velocity.random(),
													_options.angular_velocity.random(), _options.linear_damping.random(),
													_options.angular_damping.random(), _options.fading_start_time.random(),
													_options.fading_time.random());
		_numAlive++;
	}
}
