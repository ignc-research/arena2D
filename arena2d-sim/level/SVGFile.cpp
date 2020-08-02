/* author: Cornelius Marx */
#include "SVGFile.hpp"

SVGFile::SVGFile(const char * path)
{
	_path = new char[strlen(path)+1];
	strcpy(_path, path);
	_width = 0;
	_height = 0;
}

SVGFile::~SVGFile()
{
	delete[]_path;
	for(int i = 0; i < (int)_shapes.size(); i++)
		delete _shapes[i];
}


void SVGTag::print(int indents){
		printIndents(indents);
		printf("<%s", name);
		if(attribs != NULL){
			puts("");
		}
		// print all attributes
		for(SVGAttribute * it = attribs; it != NULL; it = it->next)
		{
			printIndents(indents+1);
			printf("%s=\"%s\"", it->name, it->value);
			if(it->next != NULL){
				printf("\n");
			}
		}
		printf(">\n");
		// print all children
		if(value != NULL){
			printIndents(indents);
			puts(value);
		}
		for(SVGTag * it = children; it != NULL; it = it->next){
			it->print(indents+1);
		}

		printIndents(indents);
		printf("</%s>\n", name);
}

int SVGFile::load()
{
	// load as text file
	std::string s_text;
	zStringTools::loadFromFile(_path, &s_text);
	const char * text = s_text.c_str();

	// parse XML
	SVGTag * root = parseXML(text);
	if(root == NULL){
		return 1;
	}

	// search for svg tag
	SVGTag * svg_tag = root->getChild("svg");

	int error = 0;
	if(svg_tag == NULL){
		ERROR("Could not find <svg> tag!");
		error = 1;
	}
	else{
		_width = 1;
		_height = 1;
		const char * width_str = svg_tag->getAttribute("width");
		const char * height_str = svg_tag->getAttribute("height");
		if(width_str){
			_width = atof(width_str)*SVG_SCALE;
		}
		if(height_str){
			_height = atof(height_str)*SVG_SCALE;
		}
		SVGTag * g_tag = svg_tag->getChild("g");
		if(g_tag == NULL){
			ERROR("Could not find initial <g> tag!");
			error = 1;
		}
		else{
			SVGTransform global_transform(g_tag->getAttribute("transform"));
			for(SVGTag * o = g_tag->children; o != NULL; o = o->next){
				if(!strcmp(o->name, "path")){
					SVGTransform t(o->getAttribute("transform"));
					const char *type = o->getAttribute("sodipodi:type");
					if(type != NULL && !strcmp(type, "arc"))// probably a circle
					{
						float x = 0, y = 0, r = 0.1;
						const char * atr_str;
						b2Vec2 rv(r, r);
						atr_str = o->getAttribute("sodipodi:cx");
						if(atr_str != NULL)
							x = atof(atr_str)*SVG_SCALE;
						atr_str = o->getAttribute("sodipodi:cy");
						if(atr_str != NULL)
							y = -atof(atr_str)*SVG_SCALE;
						atr_str = o->getAttribute("sodipodi:rx");
						if(atr_str != NULL)
							rv.x = atof(atr_str)*SVG_SCALE;
						atr_str = o->getAttribute("sodipodi:ry");
						if(atr_str != NULL)
							rv.y = atof(atr_str)*SVG_SCALE;

						b2CircleShape * shape = new b2CircleShape();
						t.transform(rv, 0);
						global_transform.transform(rv, 0);
						shape->m_radius = f_fmin(fabs(rv.x), fabs(rv.y));
						shape->m_p.Set(x-_width/2.f, y+_height/2.f);
						t.transform(shape->m_p, 1);
						global_transform.transform(shape->m_p, 1);
						_shapes.push_back(shape);
					}
					else{// polygon
						SVGTransform t(o->getAttribute("transform"));
						b2Vec2 pos(0,0);
						const char * d_attr = o->getAttribute("d");
						if(d_attr != NULL){
							b2Vec2 verts[9];
							int error = 0;
							int count = 0;
							const char * allowed_chars = "MmHhVvLl";
							int allowed_chars_len = 8;
							char last_char = 'M';
							float x = 0.f;
							float y = 0.f;
							float s = 0.f;
							for(; count < 9; count++){
								zStringTools::skipWhiteSpace(&d_attr);
								if(*d_attr == '\0'){
									break;
								}
								if(zStringTools::charIsElementOf(*d_attr, allowed_chars, allowed_chars_len)){
									last_char = *d_attr;
									d_attr++;
									zStringTools::skipWhiteSpace(&d_attr);
								}
								else if(*d_attr == 'z' || *d_attr == 'Z'){
									break;
								}
								else if(zStringTools::isLetter(*d_attr)){
									ERROR_F("Unsupported letter '%c' in path!", *d_attr);
									error = 1;
									break;
								}
								if(last_char == 'm' || last_char == 'M' ||
									last_char == 'l' || last_char == 'L' ){// read 2 coordinates
									x = atof(d_attr)*SVG_SCALE;
									if(zStringTools::goTo(',', &d_attr)){
										d_attr++;
										y = -atof(d_attr)*SVG_SCALE;
										verts[count] = pos;
									}
									else{
										ERROR("Expected ',' in path attribute!");
										error = 1;
										break;
									}
								}
								else{// read 1 coordinate
									s = atof(d_attr)*SVG_SCALE;
								}
								switch(last_char){
								case 'l':
								case 'm':
									pos.Set(pos.x + x, pos.y + y); break;
								case 'L':
								case 'M':
									pos.Set(x,y); break;
								case 'v':
									pos.y -= s; break;
								case 'V':
									pos.y = -s; break;
								case 'h':
									pos.x += s; break;
								case 'H':
									pos.x = s; break;
								}
								
								verts[count] = pos;

								while(!zStringTools::isWhiteSpace(*d_attr) && *d_attr != '\0'){d_attr++;}
							}
							if(count > 8){
								ERROR("Path has more than 8 verticies!");
								error = 1;
							}
							if(!error && count > 2){
								if(count > 8){
								}
								for(int i = 0; i < count; i++){
									t.transform(verts[i], 1);
									global_transform.transform(verts[i], 1);
									verts[i].x -= _width/2.f;
									verts[i].y += _height/2.f;
								}

								b2PolygonShape * shape = new b2PolygonShape();
								shape->Set(verts, count);
								_shapes.push_back(shape);
							}
						}
					}
				}
				else if(!strcmp(o->name, "rect")){
					SVGTransform t(o->getAttribute("transform"));
					b2PolygonShape * shape = new b2PolygonShape();
					b2Vec2 pos(0,0);
					float w = 0.1, h = 0.1;
					const char * attr_str;
					if((attr_str = o->getAttribute("x")))
						pos.x = atof(attr_str)*SVG_SCALE;
					if((attr_str = o->getAttribute("y")))
						pos.y = -atof(attr_str)*SVG_SCALE;
					if((attr_str = o->getAttribute("width")))
						w = atof(attr_str)*SVG_SCALE;
					if((attr_str = o->getAttribute("height")))
						h = atof(attr_str)*SVG_SCALE;
					b2Vec2 verts[4];
					verts[0].Set(pos.x, pos.y);
					verts[1].Set(pos.x, pos.y-h);
					verts[2].Set(pos.x+w, pos.y-h);
					verts[3].Set(pos.x+w, pos.y);
					for(int i = 0; i < 4; i++){
						t.transform(verts[i], 1);
						global_transform.transform(verts[i], 1);
						verts[i].x -= _width/2.f;
						verts[i].y += _height/2.f;
					}

					shape->Set(verts, 4);
					_shapes.push_back(shape);
				}
				else{
					// ignore everything else
				}
			}
		}
	}

	// free XML document
	freeXML(root);
	return error;
}

SVGTransform::SVGTransform(const char * t): matrix(b2Vec3(1,0,0),b2Vec3(0,1,0),b2Vec3(0,0,1))
{
	if(t != NULL && *t != '\0'){
		if(zStringTools::startsWith(t, "translate")){
			float x = 0, y = 0;
			if(zStringTools::goTo('(', &t)){
				t++;
				x = atof(t)*SVG_SCALE;
				if(zStringTools::goTo(',', &t)){
					t++;
					y = -atof(t)*SVG_SCALE;
				}
			}
			matrix.ez.Set(x, y, 1);
		}
		else if(zStringTools::startsWith(t, "scale")){
			float x = 0, y = 0;
			if(zStringTools::goTo('(', &t)){
				t++;
				x = atof(t);
				y = x;
				while(zStringTools::isNumber(*t) || *t == '-' || *t == '.' || *t == ' '){
					t++;
				}
				if(*t == ','){
					t++;
					y = atof(t);
				}
			}
			matrix.ex.Set(x, 0, 1);
			matrix.ey.Set(0, y, 1);
		}
		else if(zStringTools::startsWith(t, "rotate")){
			float x = 0;
			if(zStringTools::goTo('(', &t)){
				t++;
				x = -atof(t);
			}
			b2Rot r(f_rad(x));
			matrix.ex.Set(r.c, r.s, 1);
			matrix.ey.Set(-r.s, r.c, 1);
		}
	}
}

void SVGTransform::transform(b2Vec2 & v, float z)
{
	b2Vec3 v3 = b2Mul(matrix, b2Vec3(v.x, v.y, z));
	v.Set(v3.x, v3.y);
}

SVGTag* SVGFile::parseXML(const char * text)
{
	// creating root tag
	SVGTag * tag_stack[128];
	tag_stack[0] = new SVGTag("", 0);
	SVGTag * last_child_stack[128];
	last_child_stack[0] = NULL;
	int stack_index = 0;
	int line = 1;
	int error = 0;
	char buffer[512];
	char value_buffer[1024];
	int buffer_len = 0;
	int value_buffer_len = 0;
	int num_allowed_tag_chars = 3;
	const char *allowed_tag_chars = ":-_";
	while(*text != '\0'){
		line += zStringTools::skipWhiteSpace(&text);

		// parse tag
		if(*text == '<' ){
			text++;
			if(*text == '?'){// xml header
				text++;
				while(*text != '\0' && *text != '?'){
					if(*text == '\n')
						line++;
					text++;
				}
				if(!(text[0] == '?' && text[1] == '>')){
					throwParseError(END_OF_FILE, line);
					error = 1;
					break;
				}
				text+= 2;
			}
			else if(*text == '/'){// tag ends
				text++;
				// read tag name
				buffer_len = 0;
				while(zStringTools::isAlphanum(*text) ||
					zStringTools::charIsElementOf(*text, allowed_tag_chars, num_allowed_tag_chars)){
					buffer[buffer_len++] = *text;
					text++;
				}
				buffer[buffer_len] = '\0';
				if(strcmp(buffer, tag_stack[stack_index]->name) || stack_index == 0){// check whether end name is same as start name
					throwParseError(END_TAG_NEQ_START_TAG, line);
					error = 1;
					break;
				}
				if(*text != '>'){
					throwParseError(EXPECTED_CLOSE_BRACKET, line);
					error = 1;
					break;
				}
				stack_index--;
				text++;
			}
			else{// normal tag start

				// get tag name
				buffer_len = 0;
				while(zStringTools::isAlphanum(*text) ||
					zStringTools::charIsElementOf(*text, allowed_tag_chars, num_allowed_tag_chars)){
					buffer[buffer_len++] = *text;
					text++;
				}
				if(*text == '\0'){
					throwParseError(END_OF_FILE, line);
					error = 1;
					break;
				}
				if(buffer_len == 0)
				{
					throwParseError(EXPECTED_TAG, line);
					error = 1;
					break;
				}

				// add new child to current stack top element (stack index)
				SVGTag * new_tag = new SVGTag(buffer, buffer_len);
				if(last_child_stack[stack_index] == NULL){
					last_child_stack[stack_index] = new_tag;
					tag_stack[stack_index]->children = new_tag;
				}
				else{
					last_child_stack[stack_index]->next = new_tag;
					last_child_stack[stack_index] = last_child_stack[stack_index]->next;
				}

				line += zStringTools::skipWhiteSpace(&text);
				// parsing attributes
				SVGAttribute * last_attribute = NULL;
				while(*text != '\0' && *text != '>' && *text != '/'){
					buffer_len = 0;
					// get attribute name
					while(zStringTools::isAlphanum(*text) ||
						zStringTools::charIsElementOf(*text, allowed_tag_chars, num_allowed_tag_chars)){
						buffer[buffer_len++] = *text;
						text++;
					}
					if(*text == '\0'){
						throwParseError(END_OF_FILE, line);
						error = 1;
						break;
					}
					line += zStringTools::skipWhiteSpace(&text);
					if(*text != '='){
						throwParseError(EXPECTED_EQUALS, line);
						error = 1;
						break;
					}
					text++;
					line += zStringTools::skipWhiteSpace(&text);
					if(*text != '"'){
						throwParseError(EXPECTED_VALUE, line);
						error = 1;
						break;
					}
					text++;
					// get attribute value
					value_buffer_len = 0;
					while(*text != '\0' && *text != '\n' && *text != '"'){
						value_buffer[value_buffer_len++] = *text;
						text++;
					}
					if(*text == '\0'){
						throwParseError(END_OF_FILE, line);
						error = 1;
						break;
					}
					if(*text == '\n'){
						throwParseError(VALUE_LINEBREAK, line);
						error = 1;
						break;
					}
					text++;

					// adding attribute
					SVGAttribute * new_attrib = new SVGAttribute(buffer, buffer_len, value_buffer, value_buffer_len);
					if(last_attribute == NULL){
						last_attribute = new_attrib;
						last_child_stack[stack_index]->attribs = last_attribute;
					}
					else{
						last_attribute->next = new_attrib;
						last_attribute = last_attribute->next;
					}

					// go to next attribute
					line += zStringTools::skipWhiteSpace(&text);
				}
				if(error)
					break;

				if(*text == '\0'){
					throwParseError(END_OF_FILE, line);
					error = 1;
					break;
				}
				if(*text == '>'){// go inside new tag
					text++;
					stack_index++;
					last_child_stack[stack_index] = NULL;
					tag_stack[stack_index] = last_child_stack[stack_index-1];
				}
				else if(*text == '/'){// new tag ends
					text++;
					if(*text != '>'){
						throwParseError(EXPECTED_CLOSE_BRACKET, line);
						error = 1;
						break;
					}
					text++;
				}
			}
		}
		else{// other character
			value_buffer_len = 0;
			while(*text != '\0' && *text != '<'){
				value_buffer[value_buffer_len++] = *text;
				text++;
			}
			tag_stack[stack_index]->setValue(value_buffer, value_buffer_len);
		}
	}

	
	if(!error && stack_index != 0){// some tag was not closed
		throwParseError(END_OF_FILE, line);
		error = 1;
	}
	
	if(error){
		delete tag_stack[0];
		tag_stack[0] = NULL;
	}

	return tag_stack[0];
}

void SVGFile::throwParseError(XMLError e, int line)
{
	switch(e){
		case END_OF_FILE:{
			ERROR_F("Line %d: Unexpected end of file!", line);
		}break;
		case EXPECTED_TAG:{
			ERROR_F("Line %d: Expected tag or attribute name!", line);
		}break;
		case EXPECTED_EQUALS:{
			ERROR_F("Line %d: Expected '='!", line);
		}break;
		case VALUE_LINEBREAK:{
			ERROR_F("Line %d: No line break allowed in attribute value!", line);
		}break;
		case EXPECTED_OPEN_BRACKET:{
			ERROR_F("Line %d: Expected '<'!", line);
		}break;
		case EXPECTED_CLOSE_BRACKET:{
			ERROR_F("Line %d: Expected '>'!", line);
		}break;
		case END_TAG_NEQ_START_TAG:{
			ERROR_F("Line %d: No matching tag found!", line);
		}break;
		default:{
			ERROR_F("Line %d: Parse error!", line);
		}
	}
}
