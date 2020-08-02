#include "ParticleShader.hpp"

ParticleShader::ParticleShader()
{
	// calculate maximum number of particles supported according to max num of uniform locations
	GLint num_loc;
	glGetIntegerv(GL_MAX_VERTEX_UNIFORM_COMPONENTS, &num_loc);
	_maxNumParticles = (num_loc -64)/8;

	// create vertexID buffer
	GLfloat * vertex_ids = new GLfloat[_maxNumParticles*3];
	for(int i = 0; i < _maxNumParticles; i++) {
		vertex_ids[i*3+0] = 0.0f;
		vertex_ids[i*3+1] = 1.0f;
		vertex_ids[i*3+2] = 2.0f;
	}
	glGenBuffers(1, &_vertexIDBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, _vertexIDBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*_maxNumParticles*3, vertex_ids, GL_STATIC_DRAW);
	delete[] vertex_ids;

	// create triangleID buffer
	GLfloat * triangle_ids = new GLfloat[_maxNumParticles*3];
	for(int i = 0; i < _maxNumParticles; i++) {
		triangle_ids[i*3+0] =  static_cast<float>(i);
		triangle_ids[i*3+1] =  static_cast<float>(i);
		triangle_ids[i*3+2] =  static_cast<float>(i);
	}
	glGenBuffers(1, &_triangleIDBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, _triangleIDBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*_maxNumParticles*3, triangle_ids, GL_STATIC_DRAW);
	delete[] triangle_ids;
}

ParticleShader::~ParticleShader()
{
	glDeleteBuffers(1, &_vertexIDBuffer);
	glDeleteBuffers(1, &_triangleIDBuffer);
}

int ParticleShader::load()
{
	#include <particleShader.generated.h>
	std::string v_source = VERTEX_SHADER_SOURCE;
	static const std::string max_number_token = "MAX_NUMBER_PARTICLES";
	size_t start_pos = 0;
	while(1){
		start_pos = v_source.find(max_number_token, start_pos);
		if(start_pos == std::string::npos)
			break;
		std::string num = std::to_string(_maxNumParticles);
		v_source.replace(start_pos, max_number_token.length(), num);
	}
	
	if(loadProgram(v_source.c_str(), FRAGMENT_SHADER_SOURCE) == 0){
		return -1;
	}
	_vertexIDLocation = getAttribLocation("vertexID", false);
	_triangleIDLocation = getAttribLocation("triangleID", false);
	_cameraMatrixLocation = getAttribLocation("cameraMat", true);
	_projectionMatrixLocation = getAttribLocation("projMat", true);
	_triangleUniform = getAttribLocation("triangles", true);
	return 0;
}

void ParticleShader::useProgram()
{
	zShaderProgram::useProgram();

	// binding vertexID buffer
	glBindBuffer(GL_ARRAY_BUFFER, _vertexIDBuffer);
	glVertexAttribPointer(_vertexIDLocation, 1, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(_vertexIDLocation);

	// binding triangleID buffer
	glBindBuffer(GL_ARRAY_BUFFER, _triangleIDBuffer);
	glVertexAttribPointer(_triangleIDLocation, 1, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(_triangleIDLocation);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void ParticleShader::unuseProgram()
{
	glDisableVertexAttribArray(_vertexIDBuffer);
	glDisableVertexAttribArray(_triangleIDBuffer);
}
