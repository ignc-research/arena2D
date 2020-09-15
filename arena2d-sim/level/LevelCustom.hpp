#ifndef LEVELCUSTOM_H
#define LEVELCUSTOM_H

#include "Level.hpp"
#include "WandererBipedal.hpp"

#define LEVEL_CUSTOM_GOAL_SPAWN_AREA_BLOCK_SIZE 0.1 // maximum size of block when creating quad tree of goal spawn area

/* randomly generated level with static obstacles and optional dynamic obstacles */
class LevelCustom : public Level {
public:
    /* constructor
     */
    LevelCustom(const LevelDef &d, bool dynamic = false) : Level(d), _dynamic(dynamic) {}

    /* destructor
     */
    ~LevelCustom() { freeWanderers(); }

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
    /* free wanderers and clear list
     */
    void freeWanderers();

    /* if set to true, create dynamic obstacles (wanderers) in addition to static */
    bool _dynamic;

    /* list that stores all wanderers for dynamic level */
    std::list<WandererBipedal *> _wanderers;

    /* spawn area for dynamic obstacles */
    RectSpawn _dynamicSpawn;

    b2Body *
    generateRandomBodyVertical(const b2Vec2 &position, float min_radius, float max_radius, zRect *aabb);

    b2Body *
    generateRandomBodyHorizontal(const b2Vec2 &position, float min_radius, float max_radius, zRect *aabb);

};

#endif
