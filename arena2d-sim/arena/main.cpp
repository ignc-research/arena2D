#include "Arena.hpp"

int main(int argc, char ** argv)
{
	/* create and run arena */
	Arena a;
	if(a.init(argc, argv)){
		return 1;
	}
	a.run();
	a.quit();

	return 0;
}
