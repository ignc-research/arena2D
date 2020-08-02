#ifndef LIDARCAST_H
#define LIDARCAST_H

#include <box2d/box2d.h>
#include <engine/f_math.h>

class LidarCast : public b2RayCastCallback
{
public:
	LidarCast(int num_samples, float max_distance, float start_angle, float end_angle, float noise, b2Body * filter_body);

	~LidarCast();

	// raycast callback
	float ReportFixture(	b2Fixture* fixture, const b2Vec2& point,
							const b2Vec2& normal, float fraction){
		if(fixture->GetBody() != _filterBody && !fixture->IsSensor()){//ray hits collidable fixture
			_lastPoint = point;
			return fraction;
		}

		return -1.f;//continue ray because body hit was filtered out
	}

	// perform scan in given world from given start_point, sample index increases counter clock wise
	// @angle_first_sample: angle of the first ray (sample with index 0), an angle of 0degrees corresponds to the direction (1x, 0y)
	// @proportional_random_offset: if > 0 a random offset +- is applied to the sampled distance (equal distribution)
	void scan(const b2World * world, const b2Vec2 & start_point, float zero_angle);
	// getter
	const float* getDistances(){return _distances;}
	const b2Vec2* getPoints(){return &_points[1];}
	const b2Vec2* getPointsWithCenter(){return _points;}
	int getNumSamples(){return _numSamples;}
	int getClosestIndex(){return _closestPoint;}// index of closest point

	// calculates angle (in rad) of given index in _distances
	float getAngleFromIndex(int i);

	// getting index interval that cover angles from @start_angle to @end_angle
	void getIndiciesBetweenAngle(float start_angle, float end_angle, int & index_start, int & index_end);

	// laser covers full 360 degrees
	bool is360Degrees(){
		float angle_per_index = (fabs(_endAngle-_startAngle)/(_numSamples-1));
		float angle_gap = (2*M_PI-fabs(_endAngle-_startAngle));
		return  angle_per_index >= angle_gap-0.0001;}
private:
	b2Vec2 _lastPoint;
	float *_distances;
	b2Vec2 * _points;
	int _closestPoint;// index of closest point
	int _numSamples;
	float _maxDistance;
	float _startAngle;// first sample angle in rad
	float _endAngle;// last sample in rad
	float _noise;
	b2Body * _filterBody;
};

#endif
