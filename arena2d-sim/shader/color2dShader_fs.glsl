#version 130

//input from vertex shader
in vec4 out_color;

//color output
out vec4 fragment_color;

void main(void)
{
	fragment_color = out_color;
	
	
}
