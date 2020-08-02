 #version 130 

//a simple shader for rendering plain color (not textured) 2d objects

//input data
in vec3 vertex;
in vec4 color;
in vec4 color2;

//output data to fragment shader
//out vec3 out_vertex;
out vec4 out_color;
out vec4 out_color2;

//matrices
uniform mat4 projection_mat = mat4(vec4(1,0,0,0), vec4(0,1,0,0), vec4(0,0,1,0), vec4(0,0,0,1));
uniform mat4 camera_mat = mat4(vec4(1,0,0,0), vec4(0,1,0,0), vec4(0,0,1,0), vec4(0,0,0,1));
uniform mat4 modelview_mat = mat4(vec4(1,0,0,0), vec4(0,1,0,0), vec4(0,0,1,0), vec4(0,0,0,1));

void main(void)
{
	gl_Position = projection_mat * camera_mat * modelview_mat * vec4(vertex, 1);
	out_color = color;
	out_color2 = color2;
}

