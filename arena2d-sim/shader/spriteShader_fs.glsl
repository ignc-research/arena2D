#version 120

varying vec2 out_uv;
varying vec4 out_color;

uniform sampler2D tex;

void main()
{
	vec4 tex_color = texture2D(tex, vec2(out_uv.x, out_uv.y));
	gl_FragColor = tex_color*out_color;
}

