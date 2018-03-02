#makefile

testFFT: testFFT.cpp
	g++ testFFT.cpp -g -lfftw3 -lm -L/home/eramey/Programs/fftw3/lib/ -I/home/eramey/Programs/fftw3/include/ -std=c++11 -o testFFT
