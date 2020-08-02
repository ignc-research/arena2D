#version 130

//input from vertex shader
in vec4 out_color;
in vec4 out_color2;

//color output
out vec4 fragment_color;
uniform int pattern_type = 0;   //0: no pattern, 1: horizontal stripes, 2: vertical stripes, 3: diagonal stripes(lower right to upper left),
                                //4: diagonal stripes(lower left to upper right), 5: tiling
uniform int pattern_size = 4;
void main(void)
{
	fragment_color = out_color;
	if(pattern_type > 0)
	{
        switch(pattern_type)
        {
        case 1:
        {
            if(int(gl_FragCoord.y)%(pattern_size*2) >= pattern_size)
                fragment_color = out_color2;
        }break;
        case 2:
        {
            if(int(gl_FragCoord.x)%(pattern_size*2) >= pattern_size)
                fragment_color = out_color2;
        }break;
        case 3:
        {
            if(int(gl_FragCoord.y+gl_FragCoord.x)%(pattern_size*2) >= pattern_size)
                fragment_color = out_color2;
        }break;
        case 4:
        {
            if(int(gl_FragCoord.y+8092-gl_FragCoord.x)%(pattern_size*2) >= pattern_size)
                fragment_color = out_color2;
        }break;
        default:
        {
            if(int(gl_FragCoord.x)%(2*pattern_size)/pattern_size != int(gl_FragCoord.y)%(2*pattern_size)/pattern_size)
                fragment_color = out_color2;
        }break;
        }
	}
}
