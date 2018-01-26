#include <string>
#include <iostream>
#include <fftw3.h>
#include <random>
#include <complex>
#include <cstring>
#include <string.h>
#include <memory.h>

using namespace std;

int N = 65536;
int BINSIZE = 8;
int SCALE = 128;

// Prints a distribution from a histogram array
// scale changes number of stars per bin
// binSize changes range of each bin 
void printDistribution(int hist[], int binSize = 1, int scale = 1){
	for(int i=0;i<255;i+=binSize){
        int stars = 0;
		for(int j=i;j<i+binSize;++j){
			stars+= hist[j];
		}
        for(int j=0;j<stars/scale;++j){
            cout << "*";
        }
        cout << endl;
    }
}

// Performs a 1D complex-to-complex FFT
void FFT(fftw_complex in[], fftw_complex out[]){
	fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(p);
	fftw_destroy_plan(p);
}

// Performs a 1D complex-to-complex IFFT
void IFFT(fftw_complex in[], fftw_complex out[]){
	fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(p);
	fftw_destroy_plan(p);
}

int main()
{
	// distribution generator
    mt19937 gen;
	// distribution
    normal_distribution<double> distr(0, 20);
	
	// setting up arrays to hold initial data and histograms
    char dataIn[N][4] = {0};
	int hist[4][255] = {0};
	
	// loop through data array
	for(int i=0;i<4;++i){
		for(int j=0;j<N;++j){
			char num = (char)(distr(gen)); // randomly generate character
			dataIn[j][i] = num; // record in data
			hist[i][num+128]++; // add to histogram
		}
	}
	
	// Histogram titles
	string words[4] = {"X Real", "X Imaginary", "Y Real", "Y Imaginary"};
	cout << endl;
	
	// Print all histograms
	for(int i=0;i<4;++i){
		cout << words[i] << endl << endl;
		printDistribution(hist[i], BINSIZE, SCALE);
	}
	
	// complex arrays to hold initial data
	fftw_complex xVals[N];
	fftw_complex yVals[N];
	
	// get initial data in fftw_complex format
	for(int i=0;i<N;++i){
		complex<double> x(dataIn[i][0], dataIn[i][1]);
		complex<double> y(dataIn[i][2], dataIn[i][3]);
		memcpy(&xVals[i], &x, sizeof(fftw_complex));
		memcpy(&yVals[i], &y, sizeof(fftw_complex));
	}
	
	// complex arrays to hold fourier transformed data
	fftw_complex xFFT[N];
	fftw_complex yFFT[N];
	
	// complex arrays to hold IFFT output data
	fftw_complex xOut[N];
    fftw_complex yOut[N];
	
	// Perform FFT/IFFT on initial data
	FFT(xVals, xFFT);
	FFT(yVals, yFFT);
	IFFT(xFFT, xOut);
	IFFT(yFFT, yOut);
	
	// array to hold char format of output data
	char dataOut[N][4] = {0};
//	double errors[N][4] = {0};
	double max_err = 0;
	double mean_err = 0;
	
	// loop through output data
	for(int i=0;i<N;++i){
		// put values in a temporary array for loop
		double d[] = {xOut[i][0]/N, xOut[i][1]/N, yOut[i][0]/N, yOut[i][1]/N};
		// loop through values
		for(int j=0;j<4;++j){
			// round the value and scale down by N to offset scaling from IFFT
			if(d[j] > 0){
				dataOut[i][j] = (char)(d[j]+0.5);
			}
			else{
				dataOut[i][j] = (char)(d[j]-0.5);
			}
			
			// save error statistics
			double err = d[j] - (double)dataIn[i][j];
//			errors[i][j] = err;
			if(err < 0) err = -err;
			if(err > max_err) max_err = err;
			mean_err += err/(N*4.0);
			
			// check value against initial value
			if(dataIn[i][j] != dataOut[i][j]){
				// will print if data does not match
				cout << "Values do not match: " << (int)dataIn[i][j] << ", " << (int)dataOut[i][j] << endl;
			}
		}
	}
	
	// print error statistics
	cout << endl << "Mean error: " << mean_err << endl;
	cout << "Max error: " << max_err << endl << endl;
//	cout << "first 25 errors: " << endl;
//	for(int i=0;i<25;++i){
//		for(int j=0;j<4;++j){
//			cout << errors[i][j] << ", ";
//		}
//		cout << endl;
//	}
}

