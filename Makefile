mle: mle.cc
	${CXX} -O2 -std=c++11 -fopenmp -I/usr/include/eigen3 -lglog -lceres -o mle mle.cc
