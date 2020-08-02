/* author: Cornelius Marx */
#ifndef CAMERA_H
#define CAMERA_H

#include "Renderer.hpp"
#include "zVector2d.hpp"

#define DEFAULT_CAMERA_ZOOM_SPEED 1.2
#define DEFAULT_CAMERA_MAX_ZOOM 10.0
#define DEFAULT_CAMERA_MIN_ZOOM 0.05
//2D Camera
class Camera
{
public:
	Camera(): Camera(DEFAULT_CAMERA_MIN_ZOOM, DEFAULT_CAMERA_MAX_ZOOM, DEFAULT_CAMERA_ZOOM_SPEED){}
	Camera(float min_zoom, float max_zoom, float zoom_speed);
	void set(const zVector2D & pos, float zoom, float rotation){_pos = pos; setZoom(zoom); _rotation = rotation;}
	void setPos(const zVector2D & pos){_pos = pos;}
	void setRotation(float rotation){_rotation = rotation;}
	void rotate(float r){_rotation += r;}
	void setZoom(float zoom);
	void translate(const zVector2D & amount){_pos += amount;}
	void translateScaled(const zVector2D & amount);
	// exponential zooming
	void zoomExp(float exp){setZoom(_zoom*pow(_zoomSpeed, exp));}
	void zoom(float amount){setZoom(_zoom+amount);}

	// recalculate current matrix
	void refresh();

	// upload matrix to the current shader's camera matrix
	void upload()const;

	// setter
	void setZoomRange(float max_zoom, float min_zoom){_maxZoom = max_zoom; _minZoom = min_zoom;}
	void setZoomFactor(float f){_zoomSpeed = f;}

	// getter
	float getZoom(){return _zoom;}
	const zVector2D& getPos(){return _pos;} 
	float getZoomSpeed(){return _zoomSpeed;}
	float getMinZoom(){return _minZoom;}
	float getMaxZoom(){return _maxZoom;}
	float getRotation(){return _rotation;}

	const zMatrix4x4& getInverseMatrix(){return _inverseMatrix;}
	const zMatrix4x4& getMatrix(){return _matrix;}

private:
	float _zoom;
	float _rotation; // rotation in degrees
	zVector2D _pos;
	float _zoomSpeed;
	float _minZoom;
	float _maxZoom;
	zMatrix4x4 _matrix;
	zMatrix4x4 _inverseMatrix;
};

#endif
