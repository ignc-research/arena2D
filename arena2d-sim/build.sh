#!/bin/bash
echo "--- START BUILDING ---"
mkdir -p build/
cd build/
cmake ../ $@
res=$?
build=1
if [ $res -eq "0" ]; then
	make -j
	build=$?
fi
cd ../
exit $build
