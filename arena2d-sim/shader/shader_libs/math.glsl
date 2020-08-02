float dot(vec3 a, vec3 b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

float dot(vec2 a, vec2 b)
{
    return a.x*b.x + a.y*b.y;
}

float len(vec3 a)
{
	return sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
}

float len(vec2 a)
{
	return sqrt(a.x*a.x + a.y*a.y);
}

vec3 normalize(vec3 a)
{
	return a/len(a);
}

vec2 normalize(vec2 a)
{
	return a/len(a);
}
