/* author: Cornelius Marx */
#ifndef PARTICLE_EMITTER_H
#define PARTICLE_EMITTER_H

#include "Renderer.hpp"
#include "zFramework.hpp"
#include "zColor.hpp"
#include "f_math.h"

struct Range
{
	Range(){}
	Range(float _min, float _max): min(_min), max(_max){if(_min > _max){max = _min; min = _max;}}
	Range(float value): min(value), max(value){}
	void set(float _min, float _max){min = _min; max = _max; if(_min > _max){max = _min; min = _max;}}
	void set(float value){min = value; max = value;}
	float distance()const{return max-min;}
	// get random value between min and max (including)
	float random()const{return (rand()%(int)((max-min)*1000+1))/1000.0f + min;}
	float min;
	float max;
};

// emitting particles from a given point
class ParticleEmitter
{
public:
	struct ParticleEmitterOptions{
		ParticleEmitterOptions():	spawn_time(0.01), angle(0, 2*M_PI), angular_velocity(-10, 10),
									angular_damping(0), linear_velocity(2.5f, 3.f), linear_damping(0),
									scale(0.1), fading_start_time(0.5), fading_time(0.1),
									start_color(0.95, 0.9, 0.1, 1.0f), end_color(1,1,1,1){// setting default values
		}
		Range spawn_time;
		Range angle;// what direction are particles emitted
		Range angular_velocity;// how are fast are the particles spinning (rad/s)
		Range angular_damping;// rotational deacceleration of particles (rad/s/s)
		Range linear_velocity;// how fast are particles traveling
		Range linear_damping;// how fast are particles deaccelerating
		Range scale;// size of particles
		Range fading_start_time;// how long until particle starts fading (alpha blending)
		Range fading_time;// how long does particle fading take
		zColor start_color;// color to be interpolated accross lifetime
		zColor end_color;
	};
	struct ParticleOptions{
		void init(	float _angle, float _linear_velocity, float _angular_velocity,
					float _linear_damping, float _angular_damping,
					float _fading_start_time, float _fading_time){
			alive = true;
			time_alive = 0.0f;
			angle = _angle;
			linear_velocity = _linear_velocity;
			angular_velocity = _angular_velocity;
			linear_damping = _linear_damping;
			angular_damping = _angular_damping;
			fading_start_time = _fading_start_time;
			fading_time = _fading_time;
		}
		float angle;
		float linear_velocity;
		float angular_velocity;
		float linear_damping;
		float angular_damping;
		float time_alive;// how long is the particle alive
		float fading_start_time;
		float fading_time;
		bool alive;
	};
	ParticleEmitter();
	~ParticleEmitter();

	// initialize emitter, can be called multiple times
	void init(const zVector2D & pos, const ParticleEmitterOptions & options, int max_particles = 128);

	// updating particle positions
	void update();

	// render particles (particle shader must be in use)
	void render();

	// start emitting particles
	void startEmission();

	// stop emitting particles
	void endEmission();

	// kill any particles currently alive
	void killParticles();

	void setPosition(const zVector2D & pos){_pos = pos;}
	void spawnParticles(int count);
private:
	void freeParticles();
	zVector2D _pos;// emission point
	ParticleEmitterOptions _options;
	float _timeLastSpawn;// at what time was the last particle spawned
	ParticleTriangle * _particles;// triangles needed separately for fast gpu-transmission
	ParticleOptions * _particleOptions;// additional options
	int _numUsedParticles;// number of rendered particles ( starting at offset) in array
							// NOTE: this might be greater than the number of active (visible) particles due to gaps in the array
	int _numAlive;// number of particles that are truely alive
	int _usedParticlesOffset;// index of first used particle in array
	int _growDirection;// direction the block grows along; +1: increasing indicies; -1: decreasing indicies
	int _maxParticles;// capacity of particle arrays
	bool _emitting;// currently emitting particles?
	int _nextAllocationIndex; // index in array as hint for next allocation
};

#endif
