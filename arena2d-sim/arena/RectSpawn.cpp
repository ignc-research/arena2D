#include "RectSpawn.hpp"
RectSpawn::RectSpawn() : _rectAreas(NULL), _vertexBuffer(0){
}

RectSpawn::~RectSpawn(){
	clear();
}

void RectSpawn::addRect(zRect r, float margin)
{
	r.w -= margin;	
	r.h -= margin;	
	_rects.push_back(r);
}

void RectSpawn::addCheeseRect(zRect main_rect, const b2World * w, uint16 collide_mask, float margin)
{
	zRect margin_rect = main_rect;
	margin_rect.w -= margin;
	margin_rect.h -= margin;

	// perform aabb query
	_collisionCheckParams.check_collision = false;
	_collisionCheckParams.collide_mask = collide_mask;
	_collisionCheckParams.aabbs.clear();
	checkCollision(margin_rect, w);

	// add aabbs that collided with main rect
	addCheeseRect(main_rect, _collisionCheckParams.aabbs, margin);
}

void RectSpawn::addCheeseRect(zRect main_rect, const std::vector<zRect> & holes, float margin)
{
	if(holes.size() == 0){
		addRect(main_rect, margin);
		return;
	}

	// make main rect smaller by margin
	main_rect.w -= margin;
	main_rect.h -= margin;
	if(main_rect.w < 0) main_rect.w = 0;
	if(main_rect.h < 0) main_rect.h = 0;

	float * x_anchors = new float[holes.size()*2 + 2];
	float * y_anchors = new float[holes.size()*2 + 2];
	int num_x_anchors = 2;
	x_anchors[0] = main_rect.x - main_rect.w;
	x_anchors[1] = main_rect.x + main_rect.w;
	int num_y_anchors = 2;
	y_anchors[0] = main_rect.y - main_rect.h;
	y_anchors[1] = main_rect.y + main_rect.h;

	std::vector<zRect> adj_holes(holes.size());
	int adj_holes_size = 0;
	int shape_count = 0;
	for(unsigned int i = 0; i < holes.size(); i++){
		zRect r = holes[i];
		// make holes larger by margin
		r.w += margin;
		r.h += margin;
		if(!zRect::intersect(r, main_rect, &r))
			continue;

		x_anchors[num_x_anchors] = r.x - r.w;
		num_x_anchors++;
		x_anchors[num_x_anchors] = r.x + r.w;
		num_x_anchors++;

		y_anchors[num_y_anchors] = r.y - r.h;
		num_y_anchors++;
		y_anchors[num_y_anchors] = r.y + r.h;
		num_y_anchors++;
		adj_holes[adj_holes_size] = r;
		adj_holes_size++;
	}

	// sort anchors
	f_selectionSort(x_anchors, num_x_anchors);
	f_selectionSort(y_anchors, num_y_anchors);

	float e = 0.0001f;
	// adding everything but the holes
	for(int i = 0; i < num_x_anchors-1; i++){
		for(int j = 0; j < num_y_anchors-1; j++){
			zRect r;
			r.w = (x_anchors[i+1] - x_anchors[i])/2.f;
			r.h = (y_anchors[j+1] - y_anchors[j])/2.f;
			if(r.w <= e || r.h <= e)
				continue;
			r.x = (x_anchors[i+1] + x_anchors[i])/2.f;
			r.y = (y_anchors[j+1] + y_anchors[j])/2.f;

			// check whether rect r corresponds to any hole rect
			bool correspondence = false;
			for(int hole_i = 0; hole_i < adj_holes_size; hole_i++){
				if(adj_holes[hole_i].contains(r, e)){
					correspondence = true;
					break;
				}
			}
			if(correspondence)// correspondence found -> do not add rect
				continue;
			addRect(r);
			shape_count++;
		}
	}

	delete[] x_anchors;
	delete[] y_anchors;
}

void RectSpawn::clear(){
	delete[](_rectAreas);
	_rectAreas = NULL;
	_rects.clear();
	if(_vertexBuffer)
		glDeleteBuffers(1, &_vertexBuffer);
	_vertexBuffer = 0;
}

void RectSpawn::calculateArea(){
	delete[](_rectAreas);
	int rects_size = _rects.size();
	_rectAreas = new float[rects_size];
	_areaSum = 0.f;
	for(int i = 0; i < rects_size; i++){
		float a = 2*_rects[i].w*_rects[i].h;
		_rectAreas[i] = a;
		_areaSum += a;
	}

	if(_vertexBuffer)
		glDeleteBuffers(1, &_vertexBuffer);
}

void RectSpawn::render(){
	if(_vertexBuffer == 0){
		createVertexBuffer();
	}
	Z_SHADER->enableVertexArray(true);
	glBindBuffer(GL_ARRAY_BUFFER, _vertexBuffer);
	Z_SHADER->setVertexAttribPointer2D(0, 0);
	glDrawArrays(GL_TRIANGLES, 0, 12*_rects.size());
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	Z_SHADER->enableVertexArray(false);
}

void RectSpawn::createVertexBuffer(){
	glDeleteBuffers(1, &_vertexBuffer);
	int rect_size = _rects.size();
	if(rect_size == 0){// no rects to be rendered
		return;
	}
	glGenBuffers(1, &_vertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, _vertexBuffer);
	float * verts = new float[rect_size*12];
	for(int i = 0; i < rect_size; i++){
		zRect r = _rects[i];
		verts[i*12 + 0 ] = r.x-r.w; verts[i*12 + 1 ] = r.y+r.h; 
		verts[i*12 + 2 ] = r.x-r.w; verts[i*12 + 3 ] = r.y-r.h; 
		verts[i*12 + 4 ] = r.x+r.w; verts[i*12 + 5 ] = r.y+r.h; 

		verts[i*12 + 6 ] = r.x+r.w; verts[i*12 + 7 ] = r.y+r.h; 
		verts[i*12 + 8 ] = r.x-r.w; verts[i*12 + 9 ] = r.y-r.h; 
		verts[i*12 + 10] = r.x+r.w; verts[i*12 + 11] = r.y-r.h; 
	}
	glBufferData(GL_ARRAY_BUFFER, sizeof(float)*rect_size*12, verts, GL_STATIC_DRAW);
	delete[](verts);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void RectSpawn::getRandomPoint(b2Vec2 & v)
{
	assert(_rectAreas != NULL);
	if(_rects.size() == 0)
	{
		v.Set(0,0);
		return;
	}
	// get random index, bigger rects are more likely
	int i = f_randomBuckets(_rectAreas, _rects.size(), &_areaSum);

	// get random point in rect
	zRect r = _rects[i];
	v.Set(f_frandomRange(r.x-r.w, r.x+r.w), f_frandomRange(r.y-r.h, r.y+r.h));
}

void RectSpawn::addQuadTree(const zRect & area, const b2World * w, uint16 collide_mask, float max_rect_size, float margin)
{
	const bool area_changed = (area.w != area.h);
	zRect main_area = area;
	// make square
	if(main_area.w > main_area.h)
	{
		main_area.h = main_area.w;
	}
	else if(main_area.h > main_area.w)
	{
		main_area.w = main_area.h;
	}
	

	// calculate maximum number of divisions needed
	int max_capacity = 4*((int)log2(2*main_area.w/max_rect_size)+1);
	zRect * stack = new zRect[max_capacity];
	int stack_size = 0;

	// prepare collision check
	_collisionCheckParams.check_collision = true;
	_collisionCheckParams.collide_mask = collide_mask;
	_collisionCheckParams.shape_transform.SetIdentity();

	// put main rect onto stack
	stack[stack_size++] = main_area;

	float sqrt_2 = sqrt(2);
	while(stack_size > 0)
	{
		// take top stack element
		zRect top = stack[--stack_size];
	
		// check collision with top area
		zRect top_margin(top.x, top.y, top.w +margin, top.h+margin);
		/* dont use polygon shape as test shape -> slower than circle shape
		_collisionCheckParams.test_shape.SetAsBox(	top_margin.w, top_margin.h,
													b2Vec2(top_margin.x, top_margin.y), 0);
		*/
		_collisionCheckParams.test_circle_shape.m_radius = top_margin.w*sqrt_2;
		_collisionCheckParams.test_circle_shape.m_p.Set(top_margin.x, top_margin.y);
		if(!checkCollision(top_margin, w)){
			// no collisions -> add to rect list if top rect does not lie outside area
			_rects.push_back(top);
		}
		else if(top.w*2 > max_rect_size)
		{
			// divide dimensions of top rect
			const float top_quart_w = top.w/2.0f;
			const float top_quart_h = top.h/2.0f;
			assert(stack_size <= max_capacity-4);
			// adding 4 sub rects
			if(!area_changed){// area not changed -> consider all sub rects lying within original area
				stack[stack_size++].set(top.x+top_quart_w, top.y+top_quart_h, top_quart_w, top_quart_h);
				stack[stack_size++].set(top.x-top_quart_w, top.y+top_quart_h, top_quart_w, top_quart_h);
				stack[stack_size++].set(top.x-top_quart_w, top.y-top_quart_h, top_quart_w, top_quart_h);
				stack[stack_size++].set(top.x+top_quart_w, top.y-top_quart_h, top_quart_w, top_quart_h);
			}
			else{//only add rect if it intersects with originial area
				zRect sub_rect;
				sub_rect.set(top.x+top_quart_w, top.y+top_quart_h, top_quart_w, top_quart_h);
				if(zRect::intersect(sub_rect, area))
					stack[stack_size++] = sub_rect;

				sub_rect.set(top.x-top_quart_w, top.y+top_quart_h, top_quart_w, top_quart_h);
				if(zRect::intersect(sub_rect, area))
					stack[stack_size++] = sub_rect;

				sub_rect.set(top.x-top_quart_w, top.y-top_quart_h, top_quart_w, top_quart_h);
				if(zRect::intersect(sub_rect, area))
					stack[stack_size++] = sub_rect;

				sub_rect.set(top.x+top_quart_w, top.y-top_quart_h, top_quart_w, top_quart_h);
				if(zRect::intersect(sub_rect, area))
					stack[stack_size++] = sub_rect;
			}
		}
	}

	delete[] stack;
}

bool RectSpawn::checkCollision(const zRect& area, const b2World * w)
{
	b2AABB aabb;
	aabb.lowerBound.Set(area.x - area.w,
						area.y - area.h);
	aabb.upperBound.Set(area.x + area.w,
						area.y + area.h);

	_collisionCheckParams.collision = false;
	w->QueryAABB((b2QueryCallback*)this, aabb);

	return _collisionCheckParams.collision;
}

bool RectSpawn::ReportFixture(b2Fixture* fixture)
{
	// check collide mask 
	if(fixture->GetFilterData().categoryBits & _collisionCheckParams.collide_mask)
	{
		b2Shape * s = fixture->GetShape();
		const b2Transform & t = fixture->GetBody()->GetTransform();
		if(_collisionCheckParams.check_collision){
			// test all edges if shape is chain
			if(s->m_type == b2Shape::e_chain){
				int num_indicies = ((b2ChainShape*)s)->m_count;
				for(int i = 0; i < num_indicies; i++){
					if(b2TestOverlap(
						&_collisionCheckParams.test_circle_shape, 0,
						s, i,
						_collisionCheckParams.shape_transform, t)){
						_collisionCheckParams.collision = true;
						return false;
					}
				}
			}
			// test shape
			else if(b2TestOverlap(
				&_collisionCheckParams.test_circle_shape, 0,
				s, 0,
				_collisionCheckParams.shape_transform, t)){
				_collisionCheckParams.collision = true;
				return false;
			}
		}
		else{// only aabb query, store aabb of fixture shape
			b2AABB aabb;
			s->ComputeAABB(&aabb, t, 0);
			zRect r((aabb.lowerBound.x+aabb.upperBound.x)/2, (aabb.lowerBound.y+aabb.upperBound.y)/2,
					(aabb.upperBound.x-aabb.lowerBound.x)/2, (aabb.upperBound.y-aabb.lowerBound.y)/2);
			_collisionCheckParams.aabbs.push_back(r);
		}
	}
	return true;
}
