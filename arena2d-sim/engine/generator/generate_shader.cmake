set(SHADER_GENERATOR_FILES ${PROJECT_SOURCE_DIR}/engine/generator/shadergenerator.c)
add_executable(shadergenerator ${SHADER_GENERATOR_FILES})

# generating shader files
set(SHADER_NAMES
	color2d
	colorplex2d
	sprite
	text
	particle
)
set(SHADER_FILE_ENDING "Shader.generated.h")
FOREACH(N ${SHADER_NAMES})
	set(SHADER_FILES ${SHADER_FILES} "${CMAKE_BINARY_DIR}/generated/${N}${SHADER_FILE_ENDING}")
	add_custom_command(	OUTPUT "${CMAKE_BINARY_DIR}/generated/${N}${SHADER_FILE_ENDING}"
						COMMAND shadergenerator "./shader/${N}Shader_vs.glsl"
												"./shader/${N}Shader_fs.glsl"
												"${CMAKE_BINARY_DIR}/generated/${N}${SHADER_FILE_ENDING}"
						DEPENDS ${PROJECT_SOURCE_DIR}/shader/${N}Shader_vs.glsl
								${PROJECT_SOURCE_DIR}/shader/${N}Shader_fs.glsl
								shadergenerator
						WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
					)
ENDFOREACH(N)
