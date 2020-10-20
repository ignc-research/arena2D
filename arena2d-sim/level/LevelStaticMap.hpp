#ifndef LEVELSTATICMAP_H
#define LEVELSTATICMAP_H

#include "Level.hpp"
#include "WandererBipedal.hpp"
#include <ros/ros.h>
#include <ros/console.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/GetMap.h>
#include <memory>
#include "opencv2/ximgproc.hpp"
#define LEVEL_RANDOM_GOAL_SPAWN_AREA_BLOCK_SIZE 0.1 // maximum size of block when creating quad tree of goal spawn area

struct StaticMap
{
	static std::shared_ptr<nav_msgs::OccupancyGrid> m_static_map;
	// if we dont save the node handle the logging system will not work properly, maybe a better solution can be given.
	static std::unique_ptr<ros::NodeHandle> m_nh;
	static std::shared_ptr<const nav_msgs::OccupancyGrid> getMap(const std::string &static_map_service_name)
	{
		if (!static_map_service_name.empty() && !m_static_map)
		{
			m_nh = std::unique_ptr<ros::NodeHandle>(new ros::NodeHandle("arena2d_static_map_node"));
			ros::ServiceClient map_client = m_nh->serviceClient<nav_msgs::GetMap>(static_map_service_name);
			nav_msgs::GetMap getmap;
			if (map_client.call(getmap))
			{
				ROS_INFO("Got static map sucessfully!");
			}
			else
			{
				ROS_FATAL_STREAM("Failed to get the static map,please make sure the the map service with the name \"" << static_map_service_name << "\" is provided!");
				exit(-1);
			}
			m_static_map = std::make_shared<nav_msgs::OccupancyGrid>(getmap.response.map);
		}
		return m_static_map;
	}
};

class LevelStaticMap : public Level
{
public:
	LevelStaticMap(const LevelDef &d, bool dynamic);
	~LevelStaticMap(){};

	/* reset
     */
	void reset(bool robot_position_reset) override;

	/* update
     */
	void update() override;

	/* render spawn area
     * overriding to visualize spawn area for dynamic obstacles
     */
	void renderGoalSpawn() override;

private:
	void loadStaticMap(bool enable_line_approximation = true);

	/* free wanderers and clear list
	 */
	void freeWanderers();
	// the map will retain.
	void lazyclear();

	/* the ros node pointer in which the occupancy grid map was saved */
	std::shared_ptr<const nav_msgs::OccupancyGrid> _occupancygrid_ptr;

	/* if set to true, create dynamic obstacles (wanderers) in addition to static */
	bool _dynamic;

	/* list that stores all wanderers for dynamic level */
	std::list<Wanderer *> _wanderers;

	/* spawn area for dynamic obstacles */
	RectSpawn _dynamicSpawn;
	// it takes too long to calculate the spawn area for dynamic obstaticles,to save the time, it will only be done once.
	bool _init_reset;
	/* number of bodies that shoun't be removed when lazyclear is called */
	uint32 _n_non_clear_bodies;

};

#endif
