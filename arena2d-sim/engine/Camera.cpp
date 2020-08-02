/* author: Cornelius Marx */
#include "Camera.hpp"

Camera::Camera(float min_zoom, float max_zoom, float zoom_speed) :
	_zoom(min_zoom), _rotation(0.f), _pos(0,0), _zoomSpeed(zoom_speed), _minZoom(min_zoom), _maxZoom(max_zoom)
{
	refresh();
}

void Camera::setZoom(float zoom)
{
	_zoom = zoom;
	if(_zoom < _minZoom)
		_zoom = _minZoom;

	if(_zoom > _maxZoom)
		_zoom = _maxZoom;
}

void Camera::translateScaled(const zVector2D & amount)
{
	_pos += amount.getRotated(_rotation)/_zoom;
}

void Camera::refresh()
{
	_matrix.set2DCameraTransform(_pos, _zoom, _rotation);
	_inverseMatrix.setInverse2DCameraTransform(_pos, _zoom, _rotation);
}

void Camera::upload()const
{
	Z_SHADER->setCameraMatrix((const float*)(_matrix));
}

