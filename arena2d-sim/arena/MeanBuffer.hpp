#ifndef MEAN_BUFFER_H
#define MEAN_BUFFER_H
#include <SDL2/SDL_timer.h>

// storing last n arbitrary values (ring buffer) and calculates mean
class MeanBuffer
{
public:
	MeanBuffer(int n): _mean(0), _valuesStored(0), _n(n){_values = new float[_n];}
	MeanBuffer(): MeanBuffer(100){}// default 100 values
	~MeanBuffer(){delete[] _values;}
	void push(float v){_values[_valuesStored%_n] = v; _valuesStored++;}
	void calculateMean(){
		int max_it = _valuesStored;
		if(max_it > _n)
			max_it = _n;
		float sum = 0.f;
		for(int i = 0; i < max_it; i++){
			sum += _values[i];	
		}
		if(max_it == 0)
			_mean = 0;
		else
			_mean = sum/(float)max_it;
	}
	void reset(){_mean = 0; _valuesStored = 0;}

	float getMean(){return _mean;}
	const float* getValues(){return _values;}
	int getCapacity(){return _n;}

protected:
	float _mean;
	float * _values;// holding n values
	int _valuesStored;// total amount of numbers pushed
	int _n;
};

class MeanTimeBuffer : public MeanBuffer
{
public:
	MeanTimeBuffer(int n) : MeanBuffer(n), _lastTicks(0){}
	MeanTimeBuffer() : MeanBuffer(), _lastTicks(0){}
	void startTime(){_lastTicks = SDL_GetTicks();}	
	void endTime(){push((float)(SDL_GetTicks()-_lastTicks));}

private:
	Uint32 _lastTicks;
};


#endif
