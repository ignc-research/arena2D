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

option(USE_ROS  "if set to true, arena simualtor will compiled with ros \n 
					   and agent can communicate with the simulator with ros messages and \n
					   services otherwise we will only use py_AGENT " OFF)




# find packages
find_package(SDL2 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Freetype REQUIRED)
find_package(PythonLibs REQUIRED)

if(USE_ROS)
	add_compile_options("-DUSE_ROS")
	message(STATUS "USE_ROS is enabled")
	find_package(Boost)
	find_package(OpenCV 4 REQUIRED )
	find_package(catkin REQUIRED COMPONENTS
	roscpp
	roslib
	rospy
	std_msgs
	sensor_msgs
	arena2d_msgs
	nav_msgs
	message_generation
	rostest
	roslaunch)
	include_directories(${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
	message("OpenCV package is loaded from: ${OpenCV_INCLUDE_DIRS}")			
	roslaunch_add_file_check(launch)
	###################################
	## catkin specific configuration ##
	###################################
	## The catkin_package macro generates cmake config files for your package
	## Declare things to be passed to dependent projects
	## INCLUDE_DIRS: uncomment this if your package contains header files
	## LIBRARIES: libraries you create in this project that dependent projects also need
	## CATKIN_DEPENDS: catkin_packages dependent projects also need
	## DEPENDS: system dependencies of this project that dependent projects also need
	catkin_package(
		#  INCLUDE_DIRS include
		#  LIBRARIES arena2d
		CATKIN_DEPENDS # necessary for using package defined msgs and srvs.
			message_runtime
			sensor_msgs 
			nav_msgs
			arena2d_msgs
		)
	
endif()



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

# config global setting files
set(PACKAGE_PATH ${PROJECT_SOURCE_DIR})
configure_file(engine/GlobalSettings.cpp.in ${PROJECT_SOURCE_DIR}/engine/GlobalSettings.cpp)



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

# add ros related source files 
if(USE_ROS)
list(APPEND ARENA_SOURCES arena/RosNode.cpp arena/RosNode.hpp level/LevelStaticMap.hpp level/LevelStaticMap.cpp)
endif()


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
if(USE_ROS)
	list(APPEND ARENA_LINK_LIBRARIES ${catkin_LIBRARIES} ${OpenCV_LIBS})
endif()

target_link_libraries(${PROJECT_NAME} "${ARENA_LINK_LIBRARIES}" pthread ${CMAKE_DL_LIBS} box2d util)

# specify install target
install(TARGETS ${PROJECT_NAME} DESTINATION bin)
install(FILES 	./data/fonts/Bitstream_Bold.ttf
				./data/fonts/Bitstream_Regular.ttf
				DESTINATION /usr/share/fonts/TTF/)

# set C++11 standard
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)
