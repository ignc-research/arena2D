/* author: Cornelius Marx */
#ifndef SVG_FILE_H
#define SVG_FILE_H

#include <vector>
#include <arena/PhysicsWorld.hpp>
#include <arena/RectSpawn.hpp>
#include <engine/f_math.h>
#include <engine/GlobalSettings.hpp>
#include <engine/zVector2d.hpp>
#include <engine/zStringTools.hpp>
#include <arena/RectSpawn.hpp>

#define SVG_SCALE (1/1000.0)

struct SVGAttribute{
	SVGAttribute(){name = NULL; value = NULL; next = NULL;}
	SVGAttribute(const char *_name, int name_len, const char * _value, int value_len){
		name = new char[name_len+1];
		value = new char[value_len+1];
		memcpy(name, _name, name_len);
		memcpy(value, _value, value_len);
		name[name_len] = '\0';
		value[value_len] = '\0';
		next = NULL;
	}
	~SVGAttribute(){delete[]name; delete[]value; delete next;}
	char * name;
	char * value;
	SVGAttribute * next;
};

struct SVGTag{
	SVGTag(const char * _name, int name_len){
		name = new char[name_len+1];
		memcpy(name, _name, name_len);
		name[name_len] = '\0';
		attribs = NULL;
		children = NULL;
		next = NULL;
		value = NULL;
	}
	void setValue(const char * _value, int value_len){
		delete[] value;
		value = new char[value_len+1];
		memcpy(value, _value, value_len);
		value[value_len] = '\0';
	}
	~SVGTag(){
		delete[]name;
		delete attribs;
		delete[] value;
		delete next;
	}
	// returns value of given attribute or NULL if attribute does not exist
	const char* getAttribute(const char * name){
		for(SVGAttribute * a = attribs; a != NULL; a = a->next){
			if(!strcmp(a->name, name))
				return a->value;
		}
		return NULL;
	}
	// search for immediate children, returns NULL if not found
	SVGTag* getChild(const char * name){
		for(SVGTag * c = children; c != NULL; c = c->next){
			if(!strcmp(c->name, name))
				return c;
		}
		return NULL;
	}

	void print(int indents = 0);

	static void printIndents(int indents){
		for(int i = 0; i < indents; i++){
			printf("    ");
		}
	}
	char * name;
	SVGAttribute * attribs;
	char * value;
	SVGTag * children;
	SVGTag * next;
};

struct SVGTransform{
	SVGTransform(const char * t);
	SVGTransform(const b2Mat33 &m): matrix(m){}
	void transform(b2Vec2 & v, float z);
	b2Mat33 matrix;
};

/* svg file for loading custom levels */
class SVGFile
{
public:
	SVGFile(const char * path);
	~SVGFile();
	// load from file
	// returns 0 on success, 1 on error
	int load();

	// returns root
	static SVGTag* parseXML(const char * text);
	void freeXML(SVGTag * root){delete root;}

	float getWidth()const{return _width;}
	float getHeight()const{return _height;}
	void getArea(zRect & r){r.x = 0; r.y = 0; r.w = _width/2.f; r.h = _height/2.f;}
	const std::vector<b2Shape*>& getShapes(){return _shapes;}
	const char* getPath(){return _path;}
private:
	enum XMLError{END_OF_FILE, EXPECTED_TAG, EXPECTED_EQUALS, EXPECTED_VALUE, VALUE_LINEBREAK, END_TAG_NEQ_START_TAG, EXPECTED_OPEN_BRACKET, EXPECTED_CLOSE_BRACKET};
	static void throwParseError(XMLError e, int line);

	std::vector<b2Shape*> _shapes;
	float _width;
	float _height;
	char * _path;
};

#endif
