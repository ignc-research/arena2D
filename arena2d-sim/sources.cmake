set(ARENA_ADDITIONAL_SOURCES
	# add your additional .cpp/.hpp files here, e.g.
	# level/my_custom_level.cpp	
	# level/my_custom_level.hpp
)

set(ARENA_SOURCES
# main application
	arena/main.cpp
	arena/Arena.cpp
	arena/Arena_cmd.cpp
	arena/Arena_py.cpp
	arena/Arena_update.cpp
	arena/Arena_render.cpp
	arena/Arena_processEvents.cpp
	arena/Arena.hpp
	arena/Command.cpp
	arena/Command.hpp
	arena/CommandRegister.hpp
	arena/Console.cpp
	arena/Console.hpp
	arena/ConsoleParameters.cpp
	arena/ConsoleParameters.hpp
	arena/StatsDisplay.cpp
	arena/StatsDisplay.hpp
	arena/CSVWriter.cpp
	arena/CSVWriter.hpp
	arena/PhysicsWorld.cpp
	arena/PhysicsWorld.hpp
	level/Level.cpp
	level/Level.hpp
	level/LevelFactory.hpp
	level/LevelFactory.cpp
	level/LevelEmpty.cpp
	level/LevelEmpty.hpp
	level/LevelRandom.cpp
	level/LevelRandom.hpp
	level/LevelCustom.cpp
	level/LevelCustom.hpp
	level/LevelSVG.cpp
	level/LevelSVG.hpp
	level/Wanderer.cpp
	level/Wanderer.hpp
	level/WandererBipedal.cpp
	level/WandererBipedal.hpp
	level/SVGFile.hpp
	level/SVGFile.cpp
	arena/Environment.cpp
	arena/Environment.hpp
	arena/Robot.hpp
	arena/Robot.cpp
	arena/LidarCast.hpp
	arena/LidarCast.cpp
	arena/RectSpawn.cpp
	arena/RectSpawn.hpp
	arena/Evaluation.hpp
	arena/Evaluation.cpp
	level/Wanderers.hpp
	level/Wanderers.cpp
	level/LevelHuman.cpp
	level/LevelHuman.hpp
	level/LevelMaze.cpp
	level/LevelMaze.hpp

# engine
	engine/shader/Color2DShader.cpp
	engine/shader/Color2DShader.hpp
	engine/shader/Colorplex2DShader.cpp
	engine/shader/Colorplex2DShader.hpp
	engine/shader/ParticleShader.cpp
	engine/shader/ParticleShader.hpp
	engine/shader/SpriteShader.cpp
	engine/shader/SpriteShader.hpp
	engine/shader/TextShader.cpp
	engine/shader/TextShader.hpp
	engine/shader/zShaderProgram.cpp
	engine/shader/zShaderProgram.hpp
	engine/Timer.hpp
	engine/Timer.cpp
	engine/Camera.cpp
	engine/Camera.hpp
	engine/f_math.c
	engine/f_math.h
	engine/GamePadButtonCodes.hpp
	engine/GlobalSettings.cpp
	engine/GlobalSettings.hpp
	engine/hashTable.c
	engine/hashTable.h
	engine/list.h
	engine/list.c
	engine/ParticleEmitter.cpp
	engine/ParticleEmitter.hpp
	engine/Quadrangle.hpp
	engine/Renderer.cpp
	engine/Renderer.hpp
	engine/SettingsKeys.cpp
	engine/zColor.cpp
	engine/zColor.hpp
	engine/zFont.cpp
	engine/zFont.hpp
	engine/zFramework.cpp
	engine/zFramework.hpp
	engine/zGlyphMap.cpp
	engine/zGlyphMap.hpp
	engine/zGraphicObject.cpp
	engine/zGraphicObject.hpp
	engine/zLogfile.cpp
	engine/zLogfile.hpp
	engine/zMatrix4x4.cpp
	engine/zMatrix4x4.hpp
	engine/zRect.hpp
	engine/zRect.cpp
	engine/zSingleton.hpp
	engine/zStringTools.cpp
	engine/zStringTools.hpp
	engine/zTextView.cpp
	engine/zTextView.hpp
	engine/zVector2d.cpp
	engine/zVector2d.hpp
	engine/zVector3d.cpp
	engine/zVector3d.hpp
	engine/zVector4d.hpp

#settings
	settings/SettingsStructs.h
	settings/SettingsDefault.cpp

# additional libraries
	engine/glew/glew.c
	engine/glew/glew.h
	engine/glew/eglew.h
	engine/glew/glxew.h
	engine/glew/wglew.h
)
