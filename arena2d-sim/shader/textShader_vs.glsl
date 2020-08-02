#version 120

attribute vec3 vertex;
attribute vec4 color;
attribute vec2 uv;

varying vec4 out_color;
varying vec2 out_uv;
#include "shader_libs/default_matrices.glsl"
void main()
{
	gl_Position = projMat * cameraMat * modelMat * vec4(vertex, 1); 
	out_color = color;
	out_uv = uv;
}
