#version 120

attribute float vertexID;
attribute float triangleID;

varying vec4 out_color;

// NOTE: MAX-NUMBER-PARTICLES is defined at runtime (replaced by particle-shader - class)
uniform vec4 triangles[2*MAX_NUMBER_PARTICLES];

const vec2 TRIANGLE_VERTICES[3] = vec2[3](	vec2( 0.0, 		1.0),
						vec2( 0.8660254038,	-0.5),
						vec2(-0.8660254038,	-0.5));

uniform mat4 projMat;
uniform mat4 cameraMat;
void main()
{
	float cos_rad = cos(triangles[int(triangleID)*2].w);
	float sin_rad = sin(triangles[int(triangleID)*2].w);
	float scale = triangles[int(triangleID)*2].z;
	vec4 transform = triangles[int(triangleID)*2];
	mat4 t = mat4(	vec4(scale*cos_rad, -sin_rad*scale, 0, 0),
					vec4(sin_rad*scale, scale*cos_rad, 0, 0),
					vec4(0, 0, 1, 0),
					vec4(transform.x, transform.y, 0, 1));
	gl_Position = projMat * cameraMat * t * vec4(TRIANGLE_VERTICES[int(vertexID)], 0, 1);
	out_color = triangles[int(triangleID)*2 + 1];
}
