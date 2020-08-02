#!/bin/bash

# check arguments
if [ $# -ne '1' ]; then
	echo "usage: $0 <name>"
	exit 1 
fi

# check if level file already exists
hpp_file="$1.hpp"
if [ -e $hpp_file ]; then
	echo "File '$hpp_file' already exists!"
	exit 1
fi

name=$1
upper_name=${1^^}

levelstr=$'#ifndef '$upper_name$'_H\n#define '$upper_name$'_H\n\n#include \"Level.hpp\"\n\nclass '$name$' : public Level\n{\npublic:\n\t'$name$'(const LevelDef& d):Level(d){}\n\t~'$name$'(){}\n\nprivate:\n};\n\n#endif'

echo "$levelstr" > $hpp_file
