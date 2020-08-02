#version 120

attribute vec2 vertex;
attribute vec2 uv;
attribute vec4 color;

varying vec2 out_uv;
varying vec4 out_color;

#include "shader_libs/default_matrices.glsl"

void main()
{
    gl_Position = projMat * cameraMat * modelMat * vec4(vertex, 0, 1);
    out_uv = uv;
    out_color = color;
}
