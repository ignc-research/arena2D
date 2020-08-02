#include "LidarCast.hpp"

LidarCast::LidarCast(int num_samples, float max_distance, float start_angle, float end_angle, float noise, b2Body * filter_body):  _numSamples(num_samples), _maxDistance(max_distance), _startAngle(start_angle), _endAngle(end_angle), _noise(noise), _filterBody(filter_body)
{
	_distances = new float[_numSamples];
	_points = new b2Vec2[_numSamples+2];
}

LidarCast::~LidarCast()
{
	delete[] _distances;
	delete[] _points;
}

float LidarCast::getAngleFromIndex(int i)
{
	int num_gaps = _numSamples-1;
	float delta_angle = _endAngle-_startAngle;
	float f = 0;
	if(num_gaps > 0 ){
		f = static_cast<float>(i)/num_gaps;
	}
	return _startAngle + delta_angle*f;
}

void LidarCast::scan(const b2World * world, const b2Vec2 & start_point, float zero_angle)
{
	_closestPoint = -1;
	_points[0] = start_point;
	for(int i = 0; i < _numSamples; i++) {
		float angle = zero_angle + getAngleFromIndex(i);
		b2Vec2 end_point = start_point + (_maxDistance+0.001)*b2Vec2(cos(angle), sin(angle));
		_lastPoint = end_point;
		world->RayCast((b2RayCastCallback*)this, start_point, end_point);
		_distances[i] = (_lastPoint-start_point).Length();
		_points[i+1] = _lastPoint;
		if(_closestPoint < 0 || _distances[_closestPoint] > _distances[i])
			_closestPoint = i;
		if(_distances[i] > _maxDistance){ // out of range
			_distances[i] = _maxDistance;
		}
		else if(_noise != 0.f){ // apply random offset (simulation inaccuracy)
			float offset_range = _distances[i]*_noise;
			_distances[i] += offset_range*(2*f_random()-1);
			_points[i+1] = start_point + _distances[i]*b2Vec2(cos(angle), sin(angle));
		}
	}
	// last point is first point
	_points[_numSamples+1] = _points[1];
}
