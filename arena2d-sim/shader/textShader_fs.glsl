#version 120

varying vec4 out_color;
varying vec2 out_uv;

uniform sampler2D tex;
void main()
{
	gl_FragColor =  vec4(out_color.rgb, texture2D(tex, out_uv).r * out_color.a);
}
