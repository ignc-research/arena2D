#ifndef RECT_SPAWN_AREA_H
#define RECT_SPAWN_AREA_H

#include <engine/zRect.hpp>
#include <engine/Renderer.hpp>
#include <engine/f_math.h>
#include <vector>
#include <engine/zVector2d.hpp>
#include <box2d/box2d.h>



/* class represents axis aligned rectangular spawn areas */
class RectSpawn : public b2QueryCallback{
public:
	/* constructor
	 */
	RectSpawn();

	/* destructor 
	 */
	~RectSpawn();

	/* add rect to goal spawn area
	 * @param r rect to add to spawn area
	 * @param magin make rect smaller by that amount, so that an object with a radius=margin is guaranteed to spawn completely inside rect r
	 */
	void addRect(zRect r, float margin = 0.f);

	/* clear spawn area and gl vertex buffer
	 * removes all rects previously added to the area
	 */
	void clear();

	/* add a rect with rectangular cutouts (like swiss cheese) to spawn area
	 * @param main_rect rect to put holes in
	 * @param holes defines holes to put in main_rect
	 * @param margin make holes bigger, make main rect smaller by that amount
	 */
	void addCheeseRect(zRect main_rect, const std::vector<zRect> & holes, float margin = 0.f);

	void addCheeseRect(zRect main_rect, const b2World * w, uint16 collide_mask, float margin = 0.f);

	/* calculate colliding rectangles by recursively subdividing a given start area into 4 smaller sub areas
	 * rectangles that are free of collisions are added to the rects list
	 * @param area the area to start quad tree
	 * @param w Box2D world to perform collision tests in
	 * @param collide_mask only consider fixtures with specific box2d filter category
	 * @param max_rect_size quad tree is divided until the size of the area is less or equal this amount
	 * @param margin enlarge shapes by this amount when calculating collisions, so that an object with a radius=margin is guaranteed to spawn completely inside rect area
	 */
	void addQuadTree(const zRect & area, const b2World * w, uint16 collide_mask, float max_rect_size, float margin = 0.f);

	/* call this function after all rects have been added to recalculate the spawn area
	 */
	void calculateArea();

	/* debug rendering rects
	 * vertex attribute must be enabled, color has to be set in advance by caller
	 */
	void render();

	/* create vertex buffer from current rects
	 * call this function if additional rects have been added since last call to calculateArea()
	 */
	void createVertexBuffer();

	/* sample a random point in spawn area
	 * @param v is set to a random point in spawn area if any rects have been added (area > 0)
	 * NOTE: make sure to call calculateArea() once all rects have been added, this function will not work otherwise
	 */
	void getRandomPoint(b2Vec2 & v);

	/* override b2QueryCallback member function
	 */
	bool ReportFixture(b2Fixture* fixture) override;

private:
	/* perform collision check in given area and with current _collisionCheckParams
	 * @param area the area to perform aabb query in
	 * @param world Box2D world to query
	 * @return true if collision
	 */
	bool checkCollision(const zRect & area, const b2World * w);

	float * _rectAreas;
	GLuint _vertexBuffer;
	std::vector<zRect> _rects;
	float _areaSum;

	/* helper struct to make parameters accessible to ReportFixture
	 */
	struct{
		uint16 collide_mask;
		b2PolygonShape test_shape;
		b2CircleShape test_circle_shape;
		b2Transform shape_transform;
		std::vector<zRect> aabbs;
		bool check_collision;
		bool collision;
	} _collisionCheckParams;

};

#endif
