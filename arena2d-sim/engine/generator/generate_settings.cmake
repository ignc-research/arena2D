set(SETTINGS_GENERATOR_SOURCE_FILES
				${PROJECT_SOURCE_DIR}/engine/generator/settingsgenerator.c
				${PROJECT_SOURCE_DIR}/engine/list.c)

add_executable(settingsgenerator ${SETTINGS_GENERATOR_SOURCE_FILES})

set(SETTINGS_FILE "${CMAKE_BINARY_DIR}/generated/dynamicSettings.generated.cpp")

add_custom_command(OUTPUT ${SETTINGS_FILE}
					COMMAND settingsgenerator ./settings/SettingsStructs.h ${SETTINGS_FILE}
					DEPENDS ${SETTINGS_GENERATOR_SOURCE_FILES} ./settings/SettingsStructs.h
					WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
