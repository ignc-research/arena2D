cmake_minimum_required(VERSION 2.8)
project(arena2d)

message("Variable")
# avoid OpenGL warning on compilation
if(POLICY CMP0004 AND ${CMAKE_VERSION} VERSION_LESS "3.16.2")
	cmake_policy(SET CMP0004 OLD) # use legacy OpenGL drivers
endif()
if(POLICY CMP0072)
	cmake_policy(SET CMP0072 OLD) # use legacy OpenGL drivers
endif()


# find packages
find_package(SDL2 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Freetype REQUIRED)
find_package(PythonLibs 3 REQUIRED)

message(STATUS "Using python version ${PYTHONLIBS_VERSION_STRING}")

# show found libraries and include directories
set(LIBRARY_INFO_MESSAGE 	"Libraries:\n"
							"  -> sdl2:     ${SDL2_LIBRARIES}\n"
							"  -> opengl:   ${OPENGL_gl_LIBRARY}\n"
							"  -> freetype: ${FREETYPE_LIBRARIES}\n"
							"  -> python:   ${PYTHON_LIBRARIES}\n")
set(INCLUDE_INFO_MESSAGE	"Include directories:\n"
							"  -> sdl2:     ${SDL2_INCLUDE_DIRS}\n"
							"  -> freetype: ${FREETYPE_INCLUDE_DIRS}\n"
							"  -> python:   ${PYTHON_INCLUDE_DIRS}\n")
message(${INCLUDE_INFO_MESSAGE})
message(${LIBRARY_INFO_MESSAGE})

# include external cmake files for generators
include(engine/generator/generate_shader.cmake)
include(engine/generator/generate_settings.cmake)

#### generating script c strings ####
set(SCRIPTS
	"${PROJECT_SOURCE_DIR}/scripts/monitor.py"
	"${PROJECT_SOURCE_DIR}/scripts/plot.py"
)

set(SCRIPTS_BUILD_SCRIPT "${PROJECT_SOURCE_DIR}/scripts/generate_scripts.cmake")
# set file names for generated files
set(SCRIPT_C_STRING_CPP_FILE "${CMAKE_BINARY_DIR}/generated/script_sources.generated.cpp")
set(SCRIPT_C_STRING_HEADER_FILE "${CMAKE_BINARY_DIR}/generated/script_sources.generated.h")

# add cmake command to generate files containing script code as c strings
add_custom_command(	OUTPUT ${SCRIPT_C_STRING_CPP_FILE} ${SCRIPT_C_STRING_HEADER_FILE}
					COMMAND ${CMAKE_COMMAND} 	-DSCRIPTS="${SCRIPTS}"
												-DSCRIPT_C_STRING_CPP_FILE="${SCRIPT_C_STRING_CPP_FILE}"
												-DSCRIPT_C_STRING_HEADER_FILE="${SCRIPT_C_STRING_HEADER_FILE}"
												-P "${SCRIPTS_BUILD_SCRIPT}"
					DEPENDS ${SCRIPTS} "${SCRIPTS_BUILD_SCRIPT}")
####################################

# include source files listed in external cmake files
include(sources.cmake)

# adding box2d
add_subdirectory(box2d/src/)

# include directories
include_directories(
	./
	./box2d/include/
	${CMAKE_BINARY_DIR}/generated
	${SDL2_INCLUDE_DIRS}
	${FREETYPE_INCLUDE_DIRS}
	${PYTHON_INCLUDE_DIRS}
)

# creating folders to put generated files in
add_custom_target(CREATE_GEN_DIRECTORIES ${CMAKE_COMMAND} -E make_directory
										generated/
										WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# set preprocessor defines for arena based on configuration
if(${PYTHONLIBS_VERSION_STRING} VERSION_GREATER "2.9")
	message(STATUS "Compiling with define ARENA_PYTHON_VERSION_3")
	add_compile_options("-DARENA_PYTHON_VERSION_3")
endif()

# adding arena executable
add_executable(${PROJECT_NAME}
				${SCRIPT_C_STRING_CPP_FILE} 
				${SCRIPT_C_STRING_HEADER_FILE}
				${SHADER_FILES}
				${SETTINGS_FILE}
				${ARENA_SOURCES}
				${ARENA_ADDITIONAL_SOURCES})

# adding dependencies

add_dependencies(${PROJECT_NAME} CREATE_GEN_DIRECTORIES)

# specifying link libraries
set(ARENA_LINK_LIBRARIES	
	${OPENGL_gl_LIBRARY}
	${SDL2_LIBRARIES}
	${FREETYPE_LIBRARIES}
	${PYTHON_LIBRARIES}
	)

target_link_libraries(${PROJECT_NAME} "${ARENA_LINK_LIBRARIES}" pthread ${CMAKE_DL_LIBS} box2d util)

# specify install target
install(TARGETS ${PROJECT_NAME} DESTINATION bin)
install(FILES 	./data/fonts/Bitstream_Bold.ttf
				./data/fonts/Bitstream_Regular.ttf
				DESTINATION /usr/share/fonts/TTF/)

# set C++11 standard
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)
